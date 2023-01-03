'''
Extract input for MM->ML corrections from a dcd file and a pdb file,
as provided by a SOMD ABFE calculation.
'''

from openmm import app
import MDAnalysis as mda
import numpy as np
from extract_input_perses import write_pdb


def extract_input(dcd_path, pdb_path, n_snapshots=1):
    """Extract input for MM->ML corrections from a dcd file and a pdb file,
    as provided by a SOMD ABFE calculation.

    Args:
        dcd_path (str): Path to dcd file
        pdb_path (str): Path to pdb file
        n_snapshots (int, optional): Number of snapshots to extract. Default is 1.
        If 1, this is extracted from the end of the simulation, otherwise 
        evenly-spaced snapshots are extracted.
    """
    # Get topology from pdb file and use it to create MDAnalysis Universe
    topology = app.PDBFile(pdb_path).topology
    u = mda.Universe(topology, dcd_path)

    # Get list of frame indices
    frame_idxs = []
    n_frames = u.trajectory.n_frames
    if n_snapshots == 1:  # Take snapshot from end of simulation
        frame_idxs.append(n_frames - 1)
    else:
        frame_idxs = np.arange(0, n_frames, n_frames // n_snapshots)

    # Extract positions
    for i, frame_idx in enumerate(frame_idxs):
        u.trajectory[frame_idx]
        positions = u.atoms.positions
        # Write pdb
        if n_snapshots == 1:
            ofile_name = "system_endstate"
        else:
            ofile_name = f"snapshot_{i}"

        write_pdb(topology, positions, f"{ofile_name}.pdb")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcd_path", type=str, help="Path to dcd file")
    parser.add_argument("--pdb_path", type=str, help="Path to pdb file")
    parser.add_argument("--n_snapshots", type=int, default=1,
                        help="Number of snapshots to extract. Default is 1. If 1, "
                        "this is extracted from the end of the simulation, otherwise "
                        "evenly-spaced snapshots are extracted.")
    args = parser.parse_args()

    extract_input(args.dcd_path, args.pdb_path, args.n_snapshots)
