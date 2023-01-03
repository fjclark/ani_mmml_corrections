"""For given PDB, print out a given torsion"""

from openmm import app
import MDAnalysis as mda
import argparse
from repex_analysis import get_dihedral


def print_dihedral(pdb_path, atom_indices=[19,17,16,14]):
    """Print out the dihedral angle defined by atom indices

    Args:
        pdb_path (str): Path to PDB file
        atom_indices (list): Indices of atoms defining dihedral. Defaults to [19,17,16,14],
        which corresponds to the amide torsion with the TYK2 ligands.
    """

    # Get topology from pdb file
    topology = app.PDBFile(pdb_path).topology
    # Create MDAnalysis Universe to pass to get_dihedral
    u = mda.Universe(topology, pdb_path)
    dihedral = get_dihedral(*atom_indices, u)
    print(f"Dihedral defined by atom indices {atom_indices}: {dihedral:.2f} rad")


if __name__ == "__main__":
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdb_path", type=str, help="Path to PDB file")
        parser.add_argument("--atom_indices", type=int, nargs=4, default=[19,17,16,14],
                             help="Indices of atoms defining dihedral. Defaults to [19,17,16,14],"\
                             " which corresponds to the amide torsion with the TYK2 ligands.")
        args = parser.parse_args()
    
        print_dihedral(args.pdb_path, args.atom_indices)
