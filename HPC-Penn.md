# Setup

1. Get an account for the GPC2 clusters at Penn. If you don't have one, contact your advisor.

2. Log in using:
```ssh -A <PennKey>@gpc2.sas.upenn.edu```.

Some form of two-factor authentication associated with your PennKey will be required, but this can be skipped if you set up [password-less logins with SSH](https://www.strongdm.com/blog/ssh-passwordless-login) (thanks Arpit!)

3. Set up appropriate and/or useful start-up commands.

Here are some useful modules to load in your `~/.bashrc` file:

```module load gcc                                                                                               	 
module load git                                                                                 	 
module load automake/1.16.5                                                                                   	 
module load cfitsio/4.2.0                                                                                     	 
module load fftw/3.3.10                                                                                       	 
module load blas/3.11.0                                                                                       	 
module load gsl/2.7                                                                                           	 
module load lapack/3.11.0                                                                                     	 
module load slurm/current
``` 

The last line is especially important if your allocation is located on the `mathm_compute` or `sanderson_compute` partitions. Without it you won't be able to use most SLURM functionalities!

If you need `anaconda`, you can either use ```module load miniconda/22.11.1``` or ```module load anaconda/2019.10``` and put that into your `~/.bashrc`. Use the former if you don't have a reason to use the latter.

To complete set-up, run these commands:

```
source ~/.bashrc
conda init
source ~/.bashrc
```

If you would like to use OpenMPI, either add this line to the end of `~/.bashrc`, or run:

```conda activate base-mpi-py310```

or if using `anaconda/2019.10`:

```conda activate /usr/global/miniconda/22.11.1/envs/intel-mpi-py310```

If you would like to use a second Anaconda environment with different packages installed *on top* of the MPI-enabled environment, you can run:

```conda activate --stack <second env name>```

4. Get a (single) interactive node.

```
salloc -p <partition name> -N 1 --exclusive srun --pty bash
```
(subject to change)

# Running MPI

Example (needs `mpi4py` installed):
```
OMP_NUM_THREADS=8
mpirun -np 8 python -c 'from mpi4py import MPI ; comm = MPI.COMM_WORLD ; rank = comm.Get_rank() ; print(rank)'
```

This should print integers 0 to 7 inclusive, in a non-deterministic order.
This may hang or crash; we have had issues with running MPI on the clusters in the past.

# Architecture on SLURM

* Each node has 64 CPU cores with a hyperthreading factor of 2, leading to 128 virtual cores.
* To match hyperthreading to the physical configuration, use ```OMP_NUM_THREADS=2``` (verify that this works!)

# Specifications

* Mat's group: ```mathm_compute```
* Robyn's group: ```sanderson_compute```

From Kyle:
```
Switch1 partitions:
- katifori_compute: node01
- compute: node[02-06]
- titan: node[07 - 08]
- tesla: node09
- highmem: node10
- highcore: node[11-13]
- sanderson_compute: node[14 - 23]
- bhuv_gpu: node24

Switch2 partitions:
- ulloa_compute: node[25 - 28]
- empmicro_compute: node[29 - 34]
- gpc2_compute: node[35 - 42]
- mathm_compute: node[43 - 52]
- mathm_gpu: node53 (2 Nvidia A40s)


Currently, nodes 43 - 52 are dedicated to Mat's group and are configured with
2x AMD EPYC 7343 16-core/32-thread @ 3.2 - 3.9 GHz (note: this may not be correct!)
1TB RAM
1.7 TB scratch (note: list path when found)
```

Mat’s group has the following large spaces:

```
/data5 : 91 TB
/data6 : 91 TB
/data7 : 91 TB
```

Use these to store large permanent-ish shared data-sets like ACT, Planck and LSS data.

# GPC official documentation

May be outdated.

* [General documentation](https://computing.sas.upenn.edu/gpc)
* [SLURM for GPC2 usage](https://computing.sas.upenn.edu/gpc/job/slurm)

# Jupyter notebooks on GPC2 compute nodes

Moved [tutorial here](https://docs.google.com/document/d/1_4pqEbn8G8sN-gAfdxrXpL7obYb1ghIuuOSPlwJBqKc/edit?usp=sharing) (requires Penn SAS login).