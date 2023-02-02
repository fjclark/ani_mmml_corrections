# ani_mmml_corrections
Scripts for carrying out mm->ml corrections with ANI. This repo provides code to extract all required output from a Perses RBFE calculation, and to submit all runs and analyses via slurm.

To run the example, ensure that you have set up the environment from the yaml file, and installed [this branch](https://github.com/fjclark/openmmtools/tree/NNPMultistateSampler) of openmm_tools, which is a copy of [this branch](https://github.com/dominicrufa/openmmtools/tree/origin/ommml_compat), in addition to [openmm-ml](https://github.com/openmm/openmm-ml).

To submit the example with 5 lambda windows for 3 ns per window (each iteration is 1 ps), for both the bound and complex legs, run
```python
python submit_all_cor.py --mmml_dir example_input_ejm_31 --n_iter_solvent 3000 --n_iter_complex 3000 --n_states 5
```
