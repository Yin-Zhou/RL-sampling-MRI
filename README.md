## Environment setup with pytorch, on the Ginsburg hpc cluster (system is Linux, hpc management is Slurm)
### conda init
First we login to the hpc
```
ssh uni@ginsburg.rcs.columbia.edu
```
After this you should be on the login node of the hpc. Note the login node is different from a compute node, we will set up environment on the login node, then when we submit actual jobs, they are in fact run on the compute node. 
On the Ginsburg hpc (after you ssh, you get to the login node), first we want to set up conda correctly (typically need to do this for new accounts):
```
module load anaconda
conda init bash
```
Now use Ctrl + D to logout, and then login again.
### set up conda environment
Then set up a conda virtualenv, and activate it (you can give it a different name, the env name does not matter)
```
conda create -y -n env_name python=3.8
conda activate env_name
```
Install Pytorch
```
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
After you installed Pytorch, check if it works by running `python`, then `import torch` in the python interpreter. 
If Pytorch works, then either run `quit()` or Ctrl + D to exit the python interpreter. 
