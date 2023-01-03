'''
Create input files for MM-ML switching corrections based on perses output.
The input coordinates are the final coordinates from the lambda=0 or 1 coordinates
from the edge, meaning that the starting point has effectively been equilibrated.

Many functions have been copied or adapted from:
- qmlify: https://github.com/choderalab/qmlify
- openmmtools: https://github.com/dominicrufa/openmmtools/tree/ommml_compat (ommml_compat branch)
- openmm-ml: https://github.com/openmm/openmm-ml
'''

import os
import numpy as np
from openmm import app
from openmmtools.openmm_torch.repex import get_atoms_from_resname
import openmm
from openmm import app
from openeye import oechem # Required for unpickling
import pickle as pk
import shutil
import argparse
import xml.etree.ElementTree as ET

def write_pickle(object, pickle_filename):
    """
    write a pickle

    arguments
        object : object
            picklable object
        pickle_filename : str
            name of pickle
    """
    import pickle
    with open(pickle_filename, 'wb') as f:
        pickle.dump(object, f)


def deserialize_xml(xml_filename):
    """
    Load and deserialize an xml. From openmmtools.

    arguments
        xml_filename : str
            full path of the xml filename

    returns
        xml_deserialized : deserialized xml object
    """
    from simtk.openmm import XmlSerializer
    with open(xml_filename, 'r') as infile:
        xml_readable = infile.read()
    xml_deserialized = XmlSerializer.deserialize(xml_readable)
    return xml_deserialized


def serialize_xml(object, xml_filename):
    """
    Load and deserialize an xml. From openmmtools.

    arguments
        object : object
            serializable
        xml_filename : str
            full path of the xml filename
    """
    from simtk.openmm import XmlSerializer
    with open(xml_filename, 'w') as outfile:
        serial = XmlSerializer.serialize(object)
        outfile.write(serial)


def remove_constraints(system, topology, lig_pdb_name="MOL"):
    """Copy a System, removing all constraints between atoms in the ligand.
    Adapted from openmm-ml.

    Args:
        system (System): The system to copy and modify
        topology (Topology): The topology of the system
        lig_pdb_name (str): Name of the ligand in the pdb file. Defaults to "MOL".

    Returns:
        System: A newly created System object in which the constraints have been removed
        between atoms in the ligand.
    """
    lig_atoms = get_atoms_from_resname(topology, lig_pdb_name)
    atomSet = set(lig_atoms)

    # Create an XML representation of the System.
    xml = openmm.XmlSerializer.serialize(system)
    root = ET.fromstring(xml)

    # This function decides whether a bonded interaction should be removed.
    def shouldRemove(termAtoms):
        return all(a in atomSet for a in termAtoms) == True

    # Remove constraints.
    for constraints in root.findall('./Constraints'):
        for constraint in constraints.findall('Constraint'):
            constraintAtoms = [int(constraint.attrib[p]) for p in ('p1', 'p2')]
            if shouldRemove(constraintAtoms):
                constraints.remove(constraint)

    # Create a new System from it.
    return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))


def extract_sys_top(edge_outdir, phase, new_or_old, correction_outdir, factory_npz = 'out-hybrid_factory.npy.npz'):
    """
    Given a htf_factory.npz, will extract all phases, serialize systems and pickle topologies. Adapted from qmlify.

    arguments
        edge_outdir : str
            path that contains factory_npz
        phase : str
            phase to extract
        new_or_old : str
            whether the ligand is 'new' or 'old' with respect to the transformation
            on the edge
        correction_outdir : str
            path to write systems and topologies
        factory_npz : str
            .npz of perses.relative.HybridTopologyFactory
    """
    #load the npz
    npz = np.load(os.path.join(edge_outdir, factory_npz), allow_pickle=True)
    systems_dict = npz['arr_0'].item()
    if new_or_old == "new":
        topology = systems_dict[phase]._topology_proposal.new_topology
        system = systems_dict[phase]._new_system
        # Remove intra-ligand constraints
        system = remove_constraints(system, topology)
    elif new_or_old == "old":
        topology = systems_dict[phase]._topology_proposal.old_topology
        system = systems_dict[phase]._old_system
        # Remove intra-ligand constraints
        system = remove_constraints(system, topology)
    else:
        raise ValueError("new_or_old must be 'new' or 'old'")
    
    # Write system and topology to correction_outdir
    top_filename = os.path.join(correction_outdir, f"topology.pkl")
    sys_filename = os.path.join(correction_outdir, f"system.xml")
    serialize_xml(system, sys_filename)
    write_pickle(topology, top_filename)


def open_netcdf(filename):
    from netCDF4 import Dataset
    return Dataset(filename, 'r')
    

# Functions to create PDBs from positions and topologies

def write_pdb(top, pos, filename):
    """Write a pdb given topology and positions

    Args:
        top : Topology
        pos : Positions
        filename (str): Name of the pdb file to write
    """
    filename = open(filename, 'w')
    app.PDBFile.writeHeader(top, filename)
    app.PDBFile.writeModel(top, pos, filename)
    app.PDBFile.writeFooter(top, filename)


def extract_perses_repex_to_local(edge_outdir, phase, new_or_old, correction_outdir, n_snapshots=1):
    """
    Extract perses data from nonlocal directory and copy to local; extract positions, topology, and system for each phase.
    Adapted from openmmtools.

    arguments
        edge_outdir : str
            path that contains factory_npz
        phase : str
            phase to extract
        new_or_old : str
            whether the ligand is 'new' or 'old' with respect to the transformation
            on the edge
        correction_outdir : str
            path to write systems and topologies
        n_snapshots (int, optional): Number of snapshots to extract. Default is 1. If 1, this is extracted from the
        end of the simulation, otherwise evenly-spaced snapshots are extracted.
    """
    factory_npz_path = os.path.join(edge_outdir, 'out-hybrid_factory.npy.npz')
    os.system(f"cp {factory_npz_path} {os.path.join(correction_outdir, 'out-hybrid_factory.npy.npz')}")
    extract_sys_top(edge_outdir, phase, new_or_old, correction_outdir)
    npz = np.load(factory_npz_path, allow_pickle=True)
    htf = npz['arr_0'].item()

    #topology proposal
    top_proposal_filename = os.path.join(edge_outdir, f"out-topology_proposals.pkl")
    TPs = np.load(top_proposal_filename, allow_pickle=True)

    nc_checkpoint_filename = os.path.join(edge_outdir, f"out-{phase}_checkpoint.nc")
    nc_checkpoint = open_netcdf(nc_checkpoint_filename) #yank the checkpoint interval
    checkpoint_interval = nc_checkpoint.CheckpointInterval
    all_positions = nc_checkpoint.variables['positions'] #pull all of the positions
    bv = nc_checkpoint.variables['box_vectors'] #pull the box vectors
    n_iter, n_replicas, n_atom, _ = np.shape(all_positions)
    nc_out_filename = os.path.join(edge_outdir, f"out-{phase}.nc")
    nc = open_netcdf(nc_out_filename)
    
    # Get correct endstate
    if new_or_old == "old":
        endstate = ('ligandAlambda0','old',0) 
    elif new_or_old == "new":
        endstate = ('ligandBlambda1','new',n_replicas-1)
    else:
        raise ValueError("new_or_old must be 'new' or 'old'")
    lig, state, replica = endstate

    topology = getattr(TPs[f'{phase}_topology_proposal'], f'{state}_topology')
    n_atoms = topology.getNumAtoms()
    h_to_state = getattr(htf[f"{phase}"], f'_hybrid_to_{state}_map')

    # Get list of frame indices
    frame_idxs = []
    if n_snapshots == 1:
        frame_idxs.append(n_iter - 1)
    else:
        frame_idxs = np.arange(0, n_iter, n_iter // n_snapshots)
    
    # Extract positions
    for i, frame_idx in enumerate(frame_idxs):
        positions = np.zeros(shape=(n_atoms,3))
        replica_id = np.where(nc.variables['states'][frame_idx*checkpoint_interval] == replica)[0]
        pos = all_positions[frame_idx, replica_id,:,:][0]
        for hybrid, index in h_to_state.items():
            positions[index,:] = pos[hybrid]
        positions *= 10 # Convert nm to angstroms

        # Get topology and write pdb
        topology = pk.load(open(os.path.join(correction_outdir, "topology.pkl"), 'rb'))
        if n_snapshots == 1:
            ofile_name = "system_endstate"
        else:
            ofile_name = f"snapshot_{i}"

        write_pdb(topology, positions, os.path.join(correction_outdir, f"{ofile_name}.pdb"))


def extract_ligand_input(lig_name, perses_outdir=".", mmml_outdir="mmml_corrections", n_snapshots=1):
    """Extract input files for MM-ML switching corrections from perses output

    Args:
        lig_name (str) : Name of ligand for which output is extracted
        perses_outdir (str, optional): Path to perses output dir. Defaults to ".".
        mmml_outdir (str, optional): Path to mmml corrections dir. Defaults to "mmml_corrections".
        n_snapshots (int, optional): Number of snapshots to extract. Default is 1. If 1, this is extracted from the
        end of the simulation, otherwise evenly-spaced snapshots are extracted.
    """
    # Find directory with required data
    edge_out_dirs = [d for d in os.listdir(perses_outdir) if d[:4] == "edge"]

    # Take first output dir including ligand as end point
    edge_out_dir = [d for d in edge_out_dirs if lig_name in d][0]

    # Check if ligand is "old" or "new"
    new_or_old = "old"
    if lig_name == "_".join(edge_out_dir.split('_')[3:]):
        new_or_old = "new"

    for phase in ["complex", "solvent"]:
        # Make output directory
        specific_mmml_out_dir = os.path.join(mmml_outdir, lig_name, phase)

        # Extract input files
        extract_perses_repex_to_local(edge_out_dir, phase, new_or_old, specific_mmml_out_dir, n_snapshots)
        #shutil.copy(os.path.join(f"{edge_out_dir}/xml", f"{phase}-{new_or_old}-system.gz"), f"{specific_mmml_out_dir}/perses_system.gz")


def get_lig_names(perses_outdir="."):
    """Get names of all ligands in perses output

    Args:
        perses_outdir (str, optional): Path to perses output directory. Defaults to ".".

    Returns:
        list of str: List of ligand names
    """
    # Get all edge output dirs
    out_dirs = [d for d in os.listdir(perses_outdir) if d[:4] == "edge"]
    # Function to extract lig names from each edge
    dir_to_ligs = lambda x: ("_".join(x.split('_')[1:3]) , "_".join(x.split('_')[3:]))
    # Get unique lig names
    lig_names = set([l for d in out_dirs for l in dir_to_ligs(d)])

    return lig_names


def create_output_dirs(perses_outdir=".", mmml_outdir="mmml_corrections", lig_names=[]):
    """Create output directories for MM-ML switching corrections for all ligands in
    RBFE network.

    Args:
        perses_outdir (str, optional): Path to perses output directory. Defaults to ".".
        mmml_outdir (str, optional): Name of mmml output directory to create. Defaults to "mmml_corrections".
        lig_names (_type_, optional): List of ligand names. Defaults to [], resulting in the use of all ligands
        in network.
    """
    # Get lig names
    if not lig_names:
        lig_names = get_lig_names(perses_outdir)

    # Make output dirs
    for lig_name in lig_names:
        os.makedirs(f"{mmml_outdir}/{lig_name}/solvent", exist_ok=True)
        os.makedirs(f"{mmml_outdir}/{lig_name}/complex", exist_ok=True)


def extract_all_input(perses_outdir=".", mmml_outdir="mmml_corrections", lig_names=[], n_snapshots=1):
    """Extract input files for MM-ML switching corrections from perses output

    Args:
        perses_outdir (str, optional): Path to perses output dir. Defaults to ".".
        mmml_outdir (str, optional): Path to mmml corrections dir. Defaults to "mmml_corrections".
        lig_names (_type_, optional): List of ligand names. Defaults to [], resulting in the use of all ligands
        in network.
        n_snapshots (int, optional): Number of snapshots to extract. Default is 1. If 1, this is extracted from the
        end of the simulation, otherwise evenly-spaced snapshots are extracted.
    """
    # Get lig names
    if not lig_names:
        lig_names = get_lig_names(perses_outdir)

    # Create dirs for mm-ml output
    create_output_dirs(perses_outdir, mmml_outdir, lig_names)

    # Extract input files
    for lig_name in lig_names:
        print(f"Extracting input for {lig_name}")
        extract_ligand_input(lig_name, perses_outdir, mmml_outdir, n_snapshots)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perses_outdir", type=str, default=".", help="Path to perses output dir")
    parser.add_argument("--mmml_outdir", type=str, default="mmml_corrections", help="Path to mmml corrections dir")
    parser.add_argument("--lig_names", nargs='+', type=str, default=[], help="Names of ligands to extract")
    parser.add_argument("--n_snapshots", type=int, default=1, help="Number of snapshots to extract")
    args = parser.parse_args()

    extract_all_input(args.perses_outdir, args.mmml_outdir, args.lig_names, args.n_snapshots)


if __name__ == '__main__':
    main()
