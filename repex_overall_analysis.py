import glob
from unittest import skip
from repex_analysis import get_correction
from openmmtools.multistate import MultiStateSamplerAnalyzer, MultiStateReporter
import numpy as np
import os
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

# Not used in current version of script
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
        # TODO: Think about the best way to do this - 95 % C.I.s from inter-run errors? Will
        # be large error in this error with only three repeats.
        intra_run_er = np.sqrt(np.sum(np.array(cumulative_results_dict[dg_type]["er"])**2))
        inter_run_er = np.std(cumulative_results_dict[dg_type]["dg"])
        overall_results_dict[dg_type]["er"] = np.sqrt(intra_run_er**2 + inter_run_er**2)
    
    return overall_results_dict


# Not used in current version of script
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


def get_overall_results_dict(output_dirs, temp=300):
    """Perform joint analysis of results for all replicate runs in "repeat_dirs". Assumes
    that the directories are named {lig_name}_{rep_num}, where rep_num is < 10. Writes output
    to a CSV file.

    Args:
        output_dirs (list): List of paths to directories containing the outputs for each replicate run.
        temp (int, optional): Temperature in K. Defaults to 300 K.

    Returns:
        results (dict): Dictionary of results for each ligand. Has the form {lig_name: {rep_num: {"dg": dg, "er": er}, ...}, ...}.
    """
    # Create dictionary of ligand names and replicate numbers to store results
    results = {}
    for d in output_dirs:
        lig_name = d[:-2]
        rep_num = int(d[-1])
        if lig_name not in results:
            results[lig_name] = {}
        results[lig_name][rep_num] = {}

    # Get the results for each ligand
    for lig_name in results:
        logger.info(f"Analysing ligand {lig_name}")

        # Populate dict for each replicate
        for rep_no in results[lig_name]:
            dir_name = f"{lig_name}_{rep_no}"

            # Check if the simulation crashed - if so, skip
            skip_analysis = False
            for phase in ["complex", "solvent"]:
                if os.path.exists(os.path.join(dir_name, lig_name, phase, "nan-error-logs")):
                    logger.error(f"Simulation for {lig_name} in {phase} phase crashed for replicate {rep_no}. Skipping analysis.")
                    results[lig_name][rep_no] = {"dg": np.nan, "er": np.nan}
                    skip_analysis = True
            if skip_analysis:
                continue

            # Get path to complex and solvent nc files
            # TODO: Fix redundant directory structure with ligand name subdir
            com_nc_file = os.path.join(dir_name, lig_name, "complex", "repex.nc")
            sol_nc_file = os.path.join(dir_name, lig_name, "solvent", "repex.nc")

            # Get analysers
            try:
                rep_com = MultiStateReporter(com_nc_file)
                rep_sol = MultiStateReporter(sol_nc_file)
                ana_com = MultiStateSamplerAnalyzer(rep_com)
                ana_sol = MultiStateSamplerAnalyzer(rep_sol)
            # We can't read the nc file
            except AttributeError:
                logger.error(f"Simulation for {lig_name} crashed for replicate {rep_no}. Skipping analysis.")
                results[lig_name][rep_no] = {"dg": np.nan, "er": np.nan}
                continue
            

            # Get correction for a single replicate
            cor_dict = get_correction(ana_com, ana_sol, temp)
            dg = cor_dict["corr"]["dg"]
            er = cor_dict["corr"]["er"]
            results[lig_name][rep_no] = {"dg": dg, "er": er}

        # Get the overall stats, including inter- and intra-replicate error in the overall,
        # and ignoring any nans
        # TODO: Think about the best way to do this - 95 % C.I.s from inter-run errors? Will
        # be large error in this error with only three repeats.
        mean_corr = np.nanmean([results[lig_name][rep_no]["dg"] for rep_no in results[lig_name]])
        er_inter = np.nanstd([results[lig_name][rep_no]["dg"] for rep_no in results[lig_name]])
        er_intra = np.sqrt(np.nansum([results[lig_name][rep_no]["er"]**2 for rep_no in results[lig_name]]))/len(results[lig_name])
        er_overall = np.sqrt(er_inter**2 + er_intra**2)
        results[lig_name]["overall"] = {"dg": mean_corr, "er": er_overall}

    # Return the overall results
    return results

def format_dict_to_pandas(results_dict):
    """Format results dictionary into a pandas dataframe.

    Args:
        results_dict (dict): Dictionary of results for each ligand. 
        Has the form {lig_name: {rep_num: {"dg": dg, "er": er}, ...}, ...}.

    Returns:
        res_df (pandas.DataFrame): Pandas dataframe of results.
    """
    # Create a list of dicts, where each dict is a row in the dataframe
    rows = []
    for lig_name in results_dict:
        for repeat in results_dict[lig_name]:
            row = {"lig_name": lig_name, "repeat": repeat}
            row.update(results_dict[lig_name][repeat])
            rows.append(row)

    # Create the dataframe
    res_df = pd.DataFrame(rows)
    res_df.set_index(["lig_name", "repeat"], inplace=True)
    res_df.rename(columns={"dg": "Delta G (kcal / mol)", "er": "Error (kcal / mol)"}, inplace=True)
    
    return res_df


def main():
    parser = argparse.ArgumentParser(
       description="Analyse results from multiple replicate runs.")
    parser.add_argument(
       "--overall_outdir",
       type = str,
       default=".",
       help="Path to top-level directory containing the output directories for " 
       "each replicate run for each ligand. Defaults to current directory.")
    parser.add_argument(
       "--temp",
       type=int,
       default=300,
       help="Temperature in K. Defaults to 300.")
    parser.add_argument(
       "--outname",
       type=str,
       default="overall_results",
       help="Name of output file.")
    arg = parser.parse_args()

    # Get input files - assumes naming convention {lig_name}_{rep_num}
    out_dirs = [f for f in glob.glob("*_?") if "__" not in f] # Exclude __pycache__ 

    # Analyse and write output
    res_dict = get_overall_results_dict(out_dirs, arg.temp)
    res_df = format_dict_to_pandas(res_dict)
    res_df.to_csv(f"{arg.outname}.csv")

    # Print out which ligands have failed
    failed_ligs = [lig for lig in res_df.index if np.isnan(res_df.loc[lig]["Delta G (kcal / mol)"])]
    for lig in failed_ligs:
        logger.error(f"Calculation for ligand {lig[0]}, repeat {lig[1]} failed, will resubmit.") 

    # Resubmit failures
    if len(failed_ligs) > 0:
        logger.info("Resubmitting failed calculations...")
        for lig in failed_ligs:
            lig_name = lig[0]
            rep_no = lig[1]
            dir_name = f"{lig_name}_{rep_no}"
            # Save the old output
            os.system(f"cp -r {dir_name} {dir_name}_failure")
            # Clean the input
            base_cmd = f"python submit_all_cor.py --mmml_dir {dir_name} --n_iter_solvent 3000 --n_iter_complex 1000 --n_states 5"
            os.system(base_cmd + " --mode clean")
            # Resubmit
            os.system(base_cmd)

if __name__ == "__main__":
    main()
