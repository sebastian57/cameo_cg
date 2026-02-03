(comp_env) [schmidt36@jwc09n000 chemtrain-deploy]$ 
(comp_env) [schmidt36@jwc09n000 chemtrain-deploy]$ python build.py --load_gpu_pjrt_plugin
Bazel binary path: ./bazel-6.5.0-linux-x86_64
Bazel version: 6.5.0

Building XLA and installing it in the jaxlib source tree...
Traceback (most recent call last):
  File "/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build.py", line 733, in <module>
    main()
  File "/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build.py", line 716, in main
    load_pjrt_plugin_libraries(out_dir)
  File "/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/chemtrain-deploy/build.py", line 83, in load_pjrt_plugin_libraries
    raise ModuleNotFoundError("To load shared libraries, the jax_plugins "
ModuleNotFoundError: To load shared libraries, the jax_plugins namespace package must be available.

rsync -avz juwels_booster:/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/scatter3d.png .

salloc --partition=develbooster \
       --nodes=1 \
       --gres=gpu:1 \
       --time=02:00:00 \
       --account=atmlaml

salloc --partition=develbooster \
       --nodes=1 \
       --gres=gpu:4 \
       --time=02:00:00 \
       --account=cameo


salloc --partition=develbooster \
       --nodes=1 \
       --gres=gpu:4 \
       --time=00:30:00 \
       --account=cameo


salloc --partition=booster \
       --nodes=1 \
       --gres=gpu:4 \
       --time=02:00:00 \
       --account=cameo



salloc --partition=develbooster \
       --nodes=1 \
       --gres=gpu:1 \
       --time=00:10:00 \
       --account=atmlaml



salloc --partition=develbooster \
       --nodes=1 \
       --gres=gpu:1 \
       --time=00:30:00 \
       --account=atmlaml

atmlaml (instead of cameo) 

srun --cpu_bind=none --nodes=1 --pty /bin/bash -i

hostname
nvidia-smi

source ./load_modules.sh
source ./set_lammps_paths.sh
source deploy_env/bin/activate
source comp_env/bin/activate

created a new venv for compute nodes: pip install "jax[cuda12]==0.4.37"



CUDA_VISIBLE_DEVICES=0 /p/project1/cameo/schmidt36/lammps/build/lmp -echo both -in test_lammps.in


The .yaml and training files are placed in the chemtrain-deploy/external/chemtrain directory 
python -m chemtrain.train.run_training --config allegro_fm.yaml (non dev version of chemtrain)

Need to check if the pip installation of chemtrain in the comp_env venv is going to be problematic? If yes, simply uninstall and install with the -e tag for dev version
python train_fm.py allegro_fm.yaml (dev version of chemtrain)

