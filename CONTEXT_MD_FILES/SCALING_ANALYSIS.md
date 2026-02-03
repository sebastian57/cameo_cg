

Then lets do this. We use the following setup: 

CUDA_VISIBLE_DEVICES = 0 for number of nodes (1, 2, 3, 4)
CUDA_VISIBLE_DEVICES = 0,1 for number of nodes (3, 4, 5, 6)
CUDA_VISIBLE_DEVICES = 0,1,2 for number of nodes (6)
CUDA_VISIBLE_DEVICES = 0,1,2,3 for number of nodes (4, 5, 6)

This should give us the following number of devices: 1, 2, 3, 4, 6, 8, 10, 12, 16, 18, 20, 24 which gives us enough scaling information. Based on the original submission script (scripts/run_training.sh) make a modified copy that submits, using a bash loop, the jobs mentioned above, with the specified number of nodes and CUDA_VISIBLE_DEVICES. 


