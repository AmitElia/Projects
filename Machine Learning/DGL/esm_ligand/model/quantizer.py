import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, x, atom_mask):
        #x is of shape (B, L, E)
        raise NotImplementedError
    
class SonnetExponentialMovingAverage(nn.Module):
    # Copied from https://github.com/airalcorn2/vqvae-pytorch

    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".

    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average

class VectorQuantizer(Quantizer):
    # Adapted from https://github.com/airalcorn2/vqvae-pytorch
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon, with_BN = False, initialize_limit = None):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon
        if with_BN:
            self.BN = nn.BatchNorm1d(self.embedding_dim)

        # Dictionary embeddings.
        limit = initialize_limit if initialize_limit else self.num_embeddings ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x, atom_mask):
        #x is of shape (B, L, E), atom mask of shape (B, L)
        flat_x_with_pad, flat_atom_mask = x.reshape(-1, self.embedding_dim), atom_mask.view(-1) #shape (BL, E) and (BL)
        atom_indices = flat_atom_mask.nonzero().view(-1)
        flat_x = flat_x_with_pad[atom_indices] #remove padded positions
        
        if self.BN:
            flat_x = self.BN(flat_x)

        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1) #shape (BL - Npad)
        quantized_x_no_pad = F.embedding(encoding_indices, self.e_i_ts.transpose(0, 1)) #shape (BL - Npad, E)
        quantized_x = torch.zeros(flat_x_with_pad.shape).to(x.device)
        quantized_x[atom_indices] = quantized_x_no_pad
        quantized_x = quantized_x.view(x.shape)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((flat_x.detach() - quantized_x_no_pad) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((flat_x - quantized_x_no_pad.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        encoding_indices_one_hot = F.one_hot(encoding_indices, num_classes=self.num_embeddings)
        avg_probs = torch.mean(encoding_indices_one_hot.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

        return (quantized_x, dictionary_loss, commitment_loss, encoding_indices.view(flat_x.shape[0], -1), perplexity,)