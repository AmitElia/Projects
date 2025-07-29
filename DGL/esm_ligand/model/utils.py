from typing import Tuple

import torch


def find_alignment_kabsch(P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Copied from: https://gist.github.com/mjhong0708/00f72e64155c6480a6e0e3c9d3e57e18
    """Find alignment using Kabsch algorithm between two sets of points P and Q.
    Args:
    P (torch.Tensor): A tensor of shape (N, 3) representing the first set of points.
    Q (torch.Tensor): A tensor of shape (N, 3) representing the second set of points.
    Returns:
    Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor is the rotation matrix R
    and the second tensor is the translation vector t. The rotation matrix R is a tensor of shape (3, 3)
    representing the optimal rotation between the two sets of points, and the translation vector t
    is a tensor of shape (3,) representing the optimal translation between the two sets of points.
    """
    # Shift points w.r.t centroid
    centroid_P, centroid_Q = P.mean(dim=0), Q.mean(dim=0)
    P_c, Q_c = P - centroid_P, Q - centroid_Q
    # Find rotation matrix by Kabsch algorithm
    H = P_c.T @ Q_c
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # ensure right-handedness
    d = torch.sign(torch.linalg.det(V @ U.T))
    # Trick for torch.vmap
    diag_values = torch.cat(
        [
            torch.ones(1, dtype=P.dtype, device=P.device),
            torch.ones(1, dtype=P.dtype, device=P.device),
            d * torch.ones(1, dtype=P.dtype, device=P.device),
        ]
    )
    # This is only [[1,0,0],[0,1,0],[0,0,d]]
    M = torch.eye(3, dtype=P.dtype, device=P.device) * diag_values
    R = V @ M @ U.T
    # Find translation vectors
    t = centroid_Q[None, :] - (R @ centroid_P[None, :].T).T
    t = t.T
    return R.detach(), t.squeeze().detach()


def calculate_rmsd(pos: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref.
    Args:
    pos (torch.Tensor): A tensor of shape (N, 3) representing the positions of the first set of points.
    ref (torch.Tensor): A tensor of shape (N, 3) representing the positions of the second set of points.
    Returns:
    torch.Tensor: RMSD between the two sets of points.
    """
    if pos.shape[0] != ref.shape[0]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch(ref, pos)
    ref0 = (R @ ref.T).T + t
    rmsd = torch.linalg.norm(ref0 - pos, dim=1).mean()
    return rmsd

def calculate_rmsd_batch(pos: torch.Tensor, ref: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """
    pos and ref: shape (B, L, 3)
    atom_mask: shape (B, L)
    returns: shape (B, L)
    """
    length = pos.shape[0]
    rmsds = []
    for i in range(length): #its pretty hard to do it in batch so I have to implement with a for loop
        single_pos_with_pad, single_ref_with_pad, mask = pos[i], ref[i], atom_mask[i]
        indices = mask.nonzero().view(-1)
        single_pos, single_ref = single_pos_with_pad[indices], single_ref_with_pad[indices]
        rmsds.append(calculate_rmsd(single_pos, single_ref).reshape(1))
    return torch.cat(rmsds)