"""
MPI Stacking demonstration
"""

from pixell import enmap, utils as u, reproject, pointsrcs,curvedsky as cs
from orphics import maps, io,cosmology,mpi
import mpi4py
import numpy as np
import os,sys

"""
For a controlled sandbox, let's make a full sky map
with 2 arcmin pixels
"""
# define geometry
res = 2.0 * u.arcmin
shape,wcs = enmap.fullsky_geometry(res=res)

"""
 and populate it with 1000
randomly distributed sources

We will use a convenience wrapper from orphics that does the random
source generation, but you should have a look at that function.
The sources will be Gaussian with FWHM 5 arcmin.
"""

nobj = 1000
np.random.seed(100) # set seed for reproducibility
amps = np.random.uniform(0.5,1.5,size=nobj) # draw the source amplitudes
poss,omap = maps.random_source_map(shape,wcs,nobj,fwhm=10.0*u.arcmin,amps=amps)


# Let's make a high-resolution plot of this
io.hplot(omap,'omap',downgrade=2,grid=True,ticks=10)
# Open up omap.png and look at it.  I recommend the "feh" software
# to look at high-resolution images.

"""
The convenience function also gave us an array
of the positions that we can use for thumbnail extraction

We get the number of sources (should be the same as Nobj),
which sets the total number of tasks we have to do.
"""
N = poss.shape[1] 

"""
This next call is always present in my MPI programs.
It takes the N tasks and returns:
1. the MPI communicator object needed elsewhere
2. the rank of the MPI process
3. a smaller list my_tasks

If the original large task list was [0, 1, 2, ..., N-1]
this function returns a smaller list of indices
to each of the MPI processes. e.g. each process might
get
rank 0: [0, 1, 2]
rank 1: [3, 4, 5]
rank 2: [6, 7, 8]
and so on

The number of tasks put into my_tasks will depend on
the total number of MPI processes, which is not set
inside this script, but by mpirun or srun or whatever
wrapper you used to call this script.  Also, generally,
each rank/process does not need to have the same
number of elements in my_tasks.
"""
comm,rank,my_tasks = mpi.distribute(N)
print(f"Rank {rank} has {len(my_tasks)} tasks to do..." )

"""
Since reproject.thumbnails can accept a list of positions,
you could pass a slice of poss like this
"""

my_cutout = reproject.thumbnails(omap,poss.T[my_tasks],r=20.0*u.arcmin,res=2.0*u.arcmin)
twcs = my_cutout.wcs # we will need the WCS later

"""
Notice that I had to transpose poss because of the ordering that
thumbnails expects. Notice also that I passed a list my_tasks
when I indexed poss: this is known as "fancy indexing" in
numpy, if you want to read more about it.

The output of the above should be an (m,Ny,Nx)
array of thumbnails, where m is the number of
tasks in this rank/process.

We can sum these up into an (Ny,Nx) array:
"""
my_stack = my_cutout.sum(axis=0)

"""
Now comes the MPI part. Each rank/process has
its own my_stack, which is the sum of all the
cutouts it made. We can use the MPI reduce function
to collect all the my_stack arrays from all the
rank/processes, apply an operation to it
(we choose SUM from mpi4py.MPI) and then
return the result to root=0, i.e. we have chosen
rank=0 to collect the final result in the
stack variable.

"""
stack = u.reduce(my_stack, comm, root=0, op=mpi4py.MPI.SUM)
# The WCS may have been lost through all of this, so we attach it
# back from what we got from the output of thumbnails
stack = enmap.enmap(stack,twcs)

# We then proceed only if rank is zero, since that
# is the process that has the final stack
if rank==0:
    print(my_cutout.shape)
    print(stack.shape)


    # we also really only want one
    # process (here rank=0) to do any
    # further disk output

    # 
    io.plot_img(my_cutout[5],'img.png')
    io.plot_img(stack,'stack.png')

# Note that we could have used allreduce instead

stack = u.allreduce(my_stack, comm, op=mpi4py.MPI.SUM)

"""
in which case the final summed result is made available
to all MPI rank/processes, not just rank=0.


Sometimes, its not advisable to pass a whole list
of coordinates to thumbnails. For example,
you may instead want to loop through the coordinates
and only collect thumbnails if some condition is
satisified. In this case, you can do a for loop
over indices in my_tasks:
"""

my_cutouts = [] # initialize an empty list to collect cutouts
for i,task in enumerate(my_tasks):
    # i ranges from 0 to len(my_tasks)
    # but task is the correct global index that I need to use:
    my_cutout = reproject.thumbnails(omap,poss.T[task],r=20.0*u.arcmin,res=2.0*u.arcmin)
    twcs = my_cutout.wcs # we will need this later

    # If I'm satisfied that this cutout is good to include, I just append it
    my_cutouts.append(my_cutout.copy())
    # Note that if you have selection criteria that make you
    # skip some thumbnails, you should also separately
    # be keeping track of the number of accepted thumbnails
    # since that will be needed to get the right average

my_cutouts = np.asarray(my_cutouts) # convert a list of arrays to an array

"""
You can then do a similar reduce or allreduce as above, but I'll instead
demonstrate a different MPI operation: gather.

This will gather all the thumbnails instead of reducing them through
an operation. The "all" version of this will return the result
to all rank/processes.
"""

cutouts = u.allgatherv(my_cutouts,comm)

"""
You can then get the final stack through
"""

stack = cutouts.mean(axis=0)
stack = enmap.enmap(stack,twcs)

"""
Note that the above is the full stack across all
MPI processes, and I took the mean instead of the
sum.. I didn't have to keep track of the number
of accepted stamps because that is just automatically
in the fully gathered cutouts array.
"""

if rank==0:
    # I should still be careful to only do disk
    # output with only one of the processes
    io.plot_img(stack,'stack2.png')
