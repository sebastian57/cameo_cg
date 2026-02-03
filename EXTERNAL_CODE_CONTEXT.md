## Packages / code that is missing
chemtrain and chemtrain-deploy (also includes current Allegro implementation)
aggforce
lammps

## Integrating code form this repo
Need to place the cloned repo inside chemtrain-deploy/external/chemtrain
	chemtrain in the above path is the cloned chemtrain repo 
	general workflow is thus cloning chemtrain-deploy and cloning chemtrain inside a subdirectory of chemtrain-deploy

Also need to create data_prep/datasets directory with the h5 or npz dataset files

## Integrating code with LAMMPS
Need to build LAMMPS with at least the cmake command from env_setup 

This happens in chemtrain-deploy/external/chemtrain/chemtrain-deploy
The build.py script can be run as is, without using any extra flags
	This simply creates the libconnector.so file 
The cuda compatibility is more complicated: 
Need to go through chemtrain-deploy-to-LAMMPS linking process (check chemtrain-deploy documentation)
	Current workaround was to use JAX installation from python venv and copy the library file from the site-packages
	into the correct chemtrain-deploy directory (lib)
/p/project1/cameo/schmidt36/clean_booster_env/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
	Above is the path to the shared library file. It simply needs to be renamed to pjrt_plugin.xla_cuda12.so  
	Then the linker can be built using the chemtrain scripts. 
Made a modification to a connector file: 
	connector/compiler.cpp:        std::vector<std::string> platforms = {"cuda12"};
	connector/compiler.cpp:        status = module_loader->SetPlatformIndex("cuda12");
For some reason this was compatible with the JAX (from python venv) shared library file where only cuda was not? 
	It works and only requires a small change in the LAMMPS input script (cuda12 pair-style argument)

## Running stuff
Training is run from within the code from this repo
	Evaluation scripts can also be run from here
MD simulation can be run from anywhere, simply need path to LAMMPS executable (lmp)


