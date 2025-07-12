import rdkit.Chem as Chem
import numpy as np
import h5py
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def find_chiral_center(smiles: str, with_atom_index_mapping = True):
    ## given a SMILES string with or without atom index mapping, find chiral centers.
    ## Note: if there is atom index mapping in the SMILES, this is 1-indexed from SPICE2
    ## retruns:
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = False
    mol = Chem.MolFromSmiles(smiles, ps)
    if not mol:
        print(smiles)
    num_atoms = mol.GetNumAtoms()
    chiral_centers = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=False)

    chiral_mask = np.zeros(num_atoms)
    chiral_center_neighbors = np.full((num_atoms, 4), -1)
    for pair in chiral_centers:
        #chiral_centers are list of tuples like [(3, 'R'),(4, 'S')]
        chiral_atom = mol.GetAtomWithIdx(pair[0])
        map_index, neighbor_list = chiral_atom.GetAtomMapNum(), chiral_atom.GetNeighbors()
        neighbor_map_index = [neighbor.GetAtomMapNum() for neighbor in neighbor_list]
        chiral_mask[map_index - 1] = 1 if pair[0] == "R" else 2 #map index starts from 1

        if len(neighbor_map_index) < 4:
            neighbor_map_index = neighbor_map_index + [0 for _ in range(4 - len(neighbor_map_index))]
        elif len(neighbor_map_index) > 4:
            print("Warning: There is a chiral atom with more than 4 neighbors. Only taking first 4.")
            neighbor_map_index = neighbor_map_index[:4]
        chiral_center_neighbors[map_index - 1] = np.array(neighbor_map_index) - 1 #neighbor map indices were originally 1-indexed too

    return chiral_mask, chiral_center_neighbors

def get_topological_distance_matrix(smiles: str):
    #get the topological distance between any pair of atoms, and save them using their mapping number index

    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = False
    mol = Chem.MolFromSmiles(smiles, ps)

    matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    matrix_in_mapping_index = np.zeros_like(matrix)
    r, c = matrix.shape
    for i in range(r):
        for j in range(c):
            mapped_i, mapped_j = mol.GetAtomWithIdx(i).GetAtomMapNum() - 1, mol.GetAtomWithIdx(j).GetAtomMapNum() - 1
            matrix_in_mapping_index[mapped_i][mapped_j] = matrix[i][j]
    return matrix_in_mapping_index

def parse_SPICE2_datum(group: h5py.Group):
    smiles = group['smiles'][()][0]
    chiral_mask, chiral_center_neighbors = find_chiral_center(smiles)
    topological_dis_matrix = get_topological_distance_matrix(smiles)

    atomic_numbers = group['atomic_numbers'][()]
    charges = group['mbis_charges'][()]
    formation_energies = group['formation_energy'][()]
    conformations = group['conformations'][()]

    boltzmann_distribution = softmax(formation_energies)
    sample_num = 10 if len(boltzmann_distribution) > 10 else len(boltzmann_distribution)
    selected_conformer_indices = np.random.choice(list(range(len(formation_energies))), sample_num, replace=False, p = boltzmann_distribution)
    #randomly select 10 conformers based on their boltzmann distribution

    dic = {"smiles": smiles, "chiral_mask": chiral_mask, 
           "chiral_center_neighbors": chiral_center_neighbors, "atomic_numbers": atomic_numbers,
           "topological_dist": topological_dis_matrix}
    selected_data = []
    for i in selected_conformer_indices:
        output_dict = dic.copy()
        output_dict["conformation"] = conformations[i] * 0.529177 #convert Bohr to Angstrom
        output_dict["dist"] = squareform(pdist(conformations[i], metric = "euclidean"))
        output_dict["charges"] = charges[i]
        selected_data.append(output_dict)
    return selected_data, selected_conformer_indices

def parse_whole_hdf5(file_path: str):
    data = {}
    with h5py.File(file_path, 'r') as f:
        key_list = list(f.keys())
        for k in tqdm(key_list, desc="iterating over raw dataset"):
            conformation_data, indices = parse_SPICE2_datum(f[k])
            for conformation_datum, i in zip(conformation_data, indices):
                data[k + f"_{i}"] = conformation_datum
    return data

