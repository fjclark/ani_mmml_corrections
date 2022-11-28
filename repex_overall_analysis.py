from repex_analysis import get_correction
from openmmtools.multistate import MultiStateSamplerAnalyzer, MultiStateReporter
from openmmtools.openmm_torch.repex import get_atoms_from_resname
from openmm import app
import numpy as np
import os
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import seaborn as sns
from MDAnalysis.lib.distances import calc_dihedrals
import MDAnalysis as mda
import mdtraj
import logging

logger = logging.getLogger(__name__) 

def get_overall_correction(analysers_dict, temp):
    """Get overall correction - as calculated over all supplied 
    calculations.

    Args:
        analysers_dict (dict): Dictionary of MultiStateSamplerAnalyser objects. Must
        have the form {run_name: {ana_com: MultiStateSamplerAnalyser, ana_sol: MultiStateSamplerAnalyser}, ...}.
        temp (int): Temperature in K.
        output_file (str): Path to output file.
    """
    cumulative_results_dict = {"complex":{"dg": [], "er": []},
                               "solvent":{"dg": [], "er": []},
                               "corr":{"dg": [], "er": []}}
    
    for run_name in analysers_dict:
        results_dict = get_correction(analysers_dict[run_name]["ana_com"], analysers_dict[run_name]["ana_sol"], temp)
        for dg_type in results_dict:
            cumulative_results_dict[dg_type]["dg"].append(results_dict[dg_type]["dg"])
            cumulative_results_dict[dg_type]["er"].append(results_dict[dg_type]["er"])

    # Get overall results
    overall_results_dict = {}
    for dg_type in cumulative_results_dict:
        overall_results_dict[dg_type] = {}
        # Get dg as mean of all runs
        overall_results_dict[dg_type]["dg"] = np.mean(cumulative_results_dict[dg_type]["dg"])
        # Get er (SD) by combining inter- and intra- run er
        intra_run_er = np.sqrt(np.sum(np.array(cumulative_results_dict[dg_type]["er"])**2))
        inter_run_er = np.std(cumulative_results_dict[dg_type]["dg"])
        overall_results_dict[dg_type]["er"] = np.sqrt(intra_run_er**2 + inter_run_er**2)
    
    return overall_results_dict


def write_overall_correction(analysers_dict, temp, out_file):
    """Write overall correction - as calculated over all supplied 
    calculations - to file.

    Args:
        analysers_dict (dict): Dictionary of MultiStateSamplerAnalyser objects. Must
        have the form {run_name: {ana_com: MultiStateSamplerAnalyser, ana_sol: MultiStateSamplerAnalyser}, ...}.
        temp (int): Temperature in K.
        output_file (str): Path to output file.
    """
    corr_dict = get_overall_correction(analysers_dict, temp)

    with open(out_file, "w") as f:
        f.write(
            f"Correction: {corr_dict['corr']['dg']:.3f} +/- {corr_dict['corr']['er']:.3f} kcal/mol\n")
        f.write(
            f"Complex: {corr_dict['complex']['dg']:.3f} +/- {corr_dict['complex']['er']:.3f} kcal/mol\n")
        f.write(
            f"Solvent: {corr_dict['solvent']['dg']:.3f} +/- {corr_dict['solvent']['er']:.3f} kcal/mol\n")


def analyse_all(repeat_dirs, lig_names, temp, outname):
    """Perform joint analysis of results for all replicate runs in "repeat_dirs".

    Args:
        repeat_dirs (list): List of paths to directories containing replicate runs.
        lig_names (list): List of ligand names for which to perform analysis.
        temp (int): Temperature in K.
        outname (str): Name of output directory.
    """

    for lig in lig_names:

        # Dictionary to store all the analysers
        analysers_dict = {}

        # Populate dict for each replicate
        for rep in repeat_dirs:
            # Get path to complex and solvent nc files
            com_nc_file = os.path.join(rep, lig, "complex", "repex.nc")
            sol_nc_file = os.path.join(rep, lig, "solvent", "repex.nc")
            # Get analysers
            rep_com = MultiStateReporter(com_nc_file)
            rep_sol = MultiStateReporter(sol_nc_file)
            ana_com = MultiStateSamplerAnalyzer(rep_com)
            ana_sol = MultiStateSamplerAnalyzer(rep_sol)
            # Add to dict
            analysers_dict[rep] = {"ana_com": ana_com, "ana_sol": ana_sol}

            # Get the overall correction
            if not os.path.exists(outname):
                os.makedirs(outname)
            write_overall_correction(analysers_dict, temp, os.path.join(outname, f"{lig}_overall_correction.txt"))

def main():
    parser = argparse.ArgumentParser(
       description="Analyse results from multiple replicate runs.")
    parser.add_argument(
       "--repeat_dirs",
       nargs="+",
       help="Paths to directories containing replicate runs.")
    parser.add_argument(
       "--lig_names",
       nargs="+",
       help="Names of ligands for which to perform analysis.")
    parser.add_argument(
       "--temp",
       type=int,
       default=300,
       help="Temperature in K. Defaults to 300.")
    parser.add_argument(
       "--outname",
       help="Name of output directory.")
    arg = parser.parse_args()

    analyse_all(arg.repeat_dirs, arg.lig_names, arg.temp, arg.outname)


if __name__ == "__main__":
    main()



