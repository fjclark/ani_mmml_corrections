import os
from argparse import ArgumentParser
from subprocess import Popen, PIPE, STDOUT

# Get list of out_dirs


def get_outdirs(mmml_dir):
    """Find all out_dirs in mmml_dir

    Args:
        mmml_dir (str): Path to mmml_corrections directory.

    Yields:
        tuple: Tuple of ligand name and out_dir path.
    """
    for lig_name in os.listdir(mmml_dir):
        for leg in ["complex", "solvent"]:
            path = os.path.join(mmml_dir, lig_name, leg)
            yield lig_name, path


def submit_all_corr(mmml_dir, n_iter_solvent, n_iter_complex, n_states, pdb_name="system_endstate.pdb",
                    from_amber_input = False, prm7_name="system.prm7", use_alt_init_coords=False, turn_off_repex=False):
    """Submit all corrections to slurm

    Args:
        mmml_dir (str): Path to mmml_corrections directory.
        n_iter_solvent (int): Number of iterations (of 1 ps) to run 
        simulations for in the solvent leg.
        n_iter_complex (int): Number of iterations (of 1 ps) to run 
        simulations for in the complex stage.
        n_states (int): Number of linearly-spaced lambda windows
        to run.
        pdb_name (str, optional): Name of pdb file to use for parametrisation.
        from_amber_input (bool, optional): Whether to create the system from amber input. Defaults to False.
        prm7_name (str, optional): Name of prm7 7 file. Assumed to be the same for bound and free legs. Defaults to "system.prm7".
        Only required if from_amber_input is True.
        use_alt_init_coords (bool, optional): Whether to use different initial positions for each state.
        Defaults to True.
        turn_off_repex (bool, optional): Whether to turn off replica exchange. Defaults to False.
    """
    print(f"Use alt init coords: {use_alt_init_coords}")
    # Submit runs
    job_ids = []

    for lig_name, out_dir in get_outdirs(mmml_dir):
        if out_dir.split("/")[-1] == "complex":
            n_iter = n_iter_complex
        else:
            n_iter = n_iter_solvent
        cmd = f'~/Documents/research/scripts/abfe/rbatch.sh --chdir={out_dir} submit_jobs.sh {lig_name} {n_iter} {n_states} ./{pdb_name} {str(from_amber_input)} ./{prm7_name} {str(use_alt_init_coords)} {str(turn_off_repex != True)}'
        print(cmd)
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
                  stderr=STDOUT, close_fds=True)
        output = p.stdout.read()
        job_ids.append(int(output.split()[-1]))

    # Submit analysis job
    cmd = f'~/Documents/research/scripts/abfe/rbatch.sh  --dependency=afterok:{",".join([str(i) for i in job_ids])} submit_analysis.sh {mmml_dir}'
    os.system(cmd)


def clean(mmml_dir):
    """Submit all corrections to slurm

    Args:
        mmml_dir (str): Path to mmml_corrections directory.
    """
    for lig_name, out_dir in get_outdirs(mmml_dir):
        cmd = f'rm {out_dir}/repex* {out_dir}/ani_correction* {out_dir}/input_params.txt'
        print(cmd)
        os.system(cmd)


def main():
    parser = ArgumentParser()
    parser.add_argument("--mmml_dir", type=str, default="mmml_corrections",
                        help="Path to mmml_corrections directory.")
    parser.add_argument("--n_iter_solvent", type=int, default=5000,
                        help="Number of iterations (of 1 ps) to run simulations for for solvent leg.")
    parser.add_argument("--n_iter_complex", type=int, default=5000,
                        help="Number of iterations (of 1 ps) to run simulations for for complex leg.")
    parser.add_argument("--n_states", type=int, default=10,
                        help="Number of linearly-spaced lambda-windows to use.")
    parser.add_argument("--pdb_name", type=str, default="system_endstate.pdb",
                        help="Name of pdb file to use for parametrisation.")
    parser.add_argument("--from_amber_input", type=bool, default=False,
                        help="Whether to create the system from amber input.")
    parser.add_argument("--prm7_name", type=str, default="system.prm7",
                        help="Name of prm7 file. Assumed to be the same for bound and free legs."
                        " Only used if from_amber_input is True.")
    parser.add_argument("--use_alt_int_coords", type=bool, default=False,
                        help="Use different initial positions for each state.")
    parser.add_argument("--mode", type=str, default="run",
                        help="run or clean: whether to run all corrections or clean up.")
    parser.add_argument("--turn_off_repex", type=bool, default=False, help="Turn off replica exchange.")
    
    args = parser.parse_args()

    if args.mode == "run":
        print("Submitting all corrections...")
        submit_all_corr(args.mmml_dir, args.n_iter_solvent, args.n_iter_complex, args.n_states,
                        args.pdb_name, args.from_amber_input, args.prm7_name, use_alt_init_coords=args.use_alt_int_coords,
                        turn_off_repex=args.turn_off_repex)
    elif args.mode == "clean":
        print("Cleaning all corrections...")
        clean(args.mmml_dir)


if __name__ == "__main__":
    main()
