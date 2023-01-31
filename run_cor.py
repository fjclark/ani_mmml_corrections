"""Run a single MM->ANI correction"""

import torch
from argparse import ArgumentParser
import numpy as np
import sys
from rdkit import Chem
from openmm.openmm import System
from openmm import unit
from openmm.app import (
    Simulation,
    Topology,
    StateDataReporter,
    ForceField,
    PDBReporter,
    PDBFile,
    HBonds,
    Modeller,
    AmberPrmtopFile,
    AmberInpcrdFile,
    PME,
)
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
from openmmml import MLPotential
from openmm.unit import nanometer, nanometers, molar, angstrom

from openmmtools.openmm_torch.repex import (
    MixedSystemConstructor,
    RepexConstructor,
    MultistateConstructor,
    get_atoms_from_resname,
)
from tempfile import mkstemp
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def initialize_mm_forcefield(molecule):
    forcefield = ForceField("amber/protein.ff14SB.xml",
                            "amber/tip3p_standard.xml")
    if molecule is not None:
        # Ensure we use unconstrained force field

        # OFF fit to ANI2x
        #smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield='/home/finlayclark/Documents/research/mm-ml/corrections_unconstrained/xtb_ani2x_openff_1_3.offxml')

        # OFF 1.0.0
        smirnoff = SMIRNOFFTemplateGenerator(
            molecules=molecule, forcefield='openff_unconstrained-1.0.0.offxml')
        forcefield.registerTemplateGenerator(smirnoff.generator)

        # Gaff
        #gaff = GAFFTemplateGenerator(molecules=molecule, forcefield='gaff-2.11')
        # forcefield.registerTemplateGenerator(gaff.generator)

    return forcefield


def get_lig_idx(name, file):
    """Get ligand index from sdf file"""
    ligs = []
    with open(file) as f:
        for l in f:
            if l.startswith("lig"):
                ligs.append(l[4:-1])  # Remove lig and \n

    return ligs.index(name)


def get_lig_smiles(sdf_file, idx):
    """Get smiles of mol from sdf file"""
    sdfs = Chem.SDMolSupplier(sdf_file)
    smiles = Chem.MolToSmiles(sdfs[idx])
    return smiles
    

def create_mm_system_from_scratch(lig_name, pdb_path, sdfs_path):
    """Run MM->ANI corrections for a single ligand in complex or solvent leg

    Args:
        lig_name (str): Name of ligand in form "emj_31".
        pdb_path (str): Path to pdb file which will be used for parametrisation.
        sdfs_path (str): Path to file containing all ligand sdfs.

    Returns:
        system: OpenMM system object
    """
    # Get smiles of ligand
    idx = get_lig_idx(lig_name, sdfs_path)
    smiles = get_lig_smiles(sdfs_path, idx)

    # Get ligand Molecule object
    molecule = Molecule.from_smiles(smiles)

    # Create forcefield
    forcefield = initialize_mm_forcefield(molecule)

    # Create mm system
    input_file = PDBFile(pdb_path)
    modeller = Modeller(input_file.topology, input_file.positions)
    mm_system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,)

    return mm_system


def create_mm_system_from_amber(prm7_path):
    """Create an OpenMM system from amber input files.

    Args:
        prm7_path (str): Path to prm7 file.

    Returns:
        system: OpenMM system object
    """
    prmtop = AmberPrmtopFile(prm7_path)
    mm_system = prmtop.createSystem(nonbondedMethod=PME)

    return mm_system


def run_corrections(lig_name, n_iter, n_states, pdb_path, sdfs_path, from_amber_input=False,
                    prm7_path=None, use_alt_init_coords=False, repex=True):
    """Run MM->ANI corrections for a single ligand in complex or solvent leg

    Args:
        lig_name (str): Name of ligand in form "emj_31".
        n_iter (int): Number of iterations of 1 ps MCMC cycles to run for.
        n_states (int): Number of lambda windows to use.
        pdb_path (str): Path to pdb file which will be used for parametrisation.
        sdfs_path (str): Path to file containing all ligand sdfs. Only required if from_amber_input is False.
        from_amber_input (bool, optional): Whether to create the system from amber input. Defaults to False.
        prm7_path (str, optional): Path to prm7 file. Defaults to None. Must be set if from_amber_input is True.
        use_alt_init_coords (bool, optional): Whether to use different initial 
        coordinates for each state. Defaults to True.
        repex (bool, optional): Whether to use replica exchange. Defaults to True.
    """
    if from_amber_input and prm7_path is None:
        raise ValueError("Must set prm7_path if from_amber_input is True")

    # TODO: Take temperatures as argument?
    # using temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0) for i in range(n_replicas)]
#    temperatures= [t * unit.kelvin for t in [300.0, 349.588853001336, 413.2622006394437, 495.0203973723682, 600.0, 495.0203973723682, 413.2622006394437, 349.588853001336, 300.0]]
#    temperatures = [t * unit.kelvin for t in [300.0,
#                                              307.74892982405765,
#                                              316.529617667112,
#                                              326.4794405133317,
#                                              337.75406687981456,
#                                              350.529892305901,
#                                              365.0067991241227,
#                                              381.41128369038324,
#                                              400.0,
#                                              381.41128369038324,
#                                              365.0067991241227,
#                                              350.529892305901,
#                                              337.75406687981456,
#                                              326.4794405133317,
#                                              316.529617667112,
#                                              307.74892982405765,
#                                              300.0
#                                              ]]
    temperatures = [300.0 * unit.kelvin] * n_states

    # Get paths to pdbs for initial coordinates
    init_coords_pdbs = []
    if use_alt_init_coords:
        outdir = os.path.dirname(pdb_path)
        for i in range(n_states):
            init_coords_pdbs.append(os.path.join(outdir, f"snapshot_{i}.pdb"))

    else:  # Init coordinates is none, use same coords for all states
        init_coords_pdbs = [pdb_path] * n_states

    # Write out input parameters
    with open("input_params.txt", "w") as f:
        f.write(f"n_iter: {n_iter}\n")
        f.write(f"Repex: {repex}\n")
        f.write(f"n_states: {n_states}\n")
        f.write(f"temperatures: {temperatures}\n")

    # Create pure mm system
    if from_amber_input:
        mm_system = create_mm_system_from_amber(prm7_path)
    else:
        mm_system = create_mm_system_from_scratch(lig_name, pdb_path, sdfs_path)

    # create mixed mm-ml system
    input_file = PDBFile(pdb_path)
    mixed_system = MixedSystemConstructor(system=mm_system,
                                          topology=input_file.topology,
                                          removeConstraints=True).mixed_system

    if repex:
        constructor = RepexConstructor
    else:
        constructor = MultistateConstructor

    # Get sampler
    sampler = constructor(mixed_system,
                               initial_positions=[
                                   PDBFile(pdb).positions for pdb in init_coords_pdbs],
                               temperatures=temperatures,
                               n_states=n_states,
                               mcmc_moves_kwargs={'timestep': 1.0*unit.femtoseconds,
                                                  'collision_rate': 1.0/unit.picoseconds,
                                                  'n_steps': 1000,
                                                  'reassign_velocities': True},
                               sampler_kwargs={'number_of_iterations': n_iter,
                                                                'online_analysis_interval': 10,
                                                                'online_analysis_minimum_iterations': 10, },
                               storage_kwargs={'storage': f'{os.getcwd()}/repex.nc',
                                               'checkpoint_interval': 5}
                               ).sampler

    # minimise positions at each state
    print("Minimising...")
    sampler.minimize()
    # run the sampler for (default 5000) steps
    print("Running...")
    sampler.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("--lig_name", type=str,
                        help="Name of ligand in form 'emj_31'")
    parser.add_argument("--n_iter", type=int, default=5000,
                        help="Number of iterations of 1 ps MCMC cycles to run for")
    parser.add_argument("--n_states", type=int, default=10,
                        help="Number of lambda windows to use")
    parser.add_argument("--pdb_path", type=str,
                        default="./system_endstate.pdb", help="Path to pdb file")
    parser.add_argument("--sdfs_path", type=str, default="../../../ligands.sdf",
                        help="Path to file containing all ligand sdfs. Only used if from_amber_input is False")
    parser.add_argument("--from_amber_input", type=str, default="False",
                        help="Whether or not to create system from amber input")
    parser.add_argument("--prm7_path", type=str, default=None,
                        help="Path to prmtop file. Must be set if from_amber_input is True")
    parser.add_argument("--use_alt_int_coords", type=str, default="False",
                        help="Whether or not to start replicas from different initial coordinates")
    parser.add_argument("--repex", type=str, default="False", help="Whether or not to use replica exchange")
    args = parser.parse_args()

    if args.use_alt_int_coords == "True":
        use_alt_init_coords = True
    else:
        use_alt_init_coords = False
    if args.from_amber_input == "True":
        from_amber_input = True
    else:
        from_amber_input = False
    if args.repex == "True":
        repex = True
    else:
        repex = False

    run_corrections(args.lig_name, args.n_iter, args.n_states,
                    args.pdb_path, args.sdfs_path, from_amber_input,
                    args.prm7_path, use_alt_init_coords, repex)

if __name__ == "__main__":
    main()
