# Data handling and processing utilities for the ligand dataset
"""
Datasets of protein-ligand structures - PDBbind, PDB
Proof of concept: small set of pdb files, each with resolved ligand 
	(non-covalently bound small molecule) and a single protein chain.


"""
from xml.parsers.expat import model
from Bio.PDB import PDBParser, is_aa, PPBuilder
import os
import openbabel
import plip
from plip.structure.preparation import PDBComplex



def plip_parser(file_path):
	"""
	Parse a PDB file using PLIP to extract ligand information.
	"""
	complex = PDBComplex()
	complex.load_pdb(file_path)
	first_bsid = ":".join([complex.ligands[0].hetid, complex.ligands[0].chain, str(complex.ligands[0].position)])
	print(first_bsid)
	for ligand in complex.ligands:
		if ':'.join([ligand.hetid, ligand.chain, str(ligand.position)]) == first_bsid:
			complex.characterize_complex(ligand)
	structure = complex.interaction_sets[first_bsid]

	itypes = {
		"hbonds": {hbond.resnr for hbond in (structure.hbonds_pdon + structure.hbonds_ldon)},
		"saltbridges": {saltbridge.resnr for saltbridge in (structure.saltbridge_pneg + structure.saltbridge_lneg)},
		"pications": {cation.resnr for cation in structure.pication_paro + structure.pication_laro},
		"pistacking": {pistack.resnr for pistack in structure.pistacking},
		"halogenbonds": {halogen.resnr for halogen in structure.halogen_bonds},
		"hydrophobic": {hydrophobic.resnr for hydrophobic in structure.hydrophobic_contacts},
		"waterbridges": {waterbridge.resnr for waterbridge in structure.water_bridges},
		"metalcomplexes": {metal.resnr for metal in structure.metal_complexes}
	}

	for k,i in itypes.items():
		print(f"{k}: {i}")

def parse_structure_file(file_path):
	"""
	parse a PDB file and extract: - The protein’s
	amino acid sequence. - 3D coordinates of the protein’s structure.
	3D coordinates and atom types of the ligand.
	"""
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure('ligand_structure', file_path)
	data = {
		'sample_id': os.path.basename(file_path).split('.')[0],
		'protein_sequence': '',
		'protein_coordinates': [],
		'ligand_atoms': [],
	}
	model = next(structure.get_models()) # first model in PDB
	chain = next(model.get_chains())
	#seq1 = str(PPBuilder().build_peptides(chain)[0].get_sequence())
	#seq2 = "".join([res.get_resname() for res in chain.get_residues() if is_aa(res, standard=True)])
	#print(f"Sequence from PPBuilder: {seq1}")
	#print(f"Sequence from chain.get_residues: {seq2}")
	data['protein_sequence'] = str(PPBuilder().build_peptides(chain)[0].get_sequence()) 
	data['protein_coordinates'] = [{"N": residue['N'].get_coord(), 
								 "CA": residue['CA'].get_coord(), 
								 "C": residue['C'].get_coord()} 
								 for residue in chain if is_aa(residue, standard=True)]
	for res in model.get_residues():
		if res.id[0] != " ":
			if res.get_resname() not in ("HOH", "WAT"):
				for atom in res.get_atoms():
					data['ligand_atoms'].append((atom.get_id(), atom.get_coord()))
	return (data['sample_id'],
			data['protein_sequence'],
			data['protein_coordinates'],
			data['ligand_atoms'])

def main():
    print("This code runs when the script is executed directly.")
    # Add other functions or logic here
    file_path = os.path.join(os.path.dirname(__file__), "sample", "2w0s.pdb")
    sample_id, protein_sequence, protein_coords, ligand_atoms = parse_structure_file(file_path)
    print(sample_id)
    print("seq:", protein_sequence)
    print("coords:", protein_coords[:5])
    print("ligand:", ligand_atoms[:5])
    plip_parser(file_path)

if __name__ == "__main__":
    main()
