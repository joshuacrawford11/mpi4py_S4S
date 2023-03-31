"""
A simple example of scattering and gathering arrays using mpi4py.

We will scatter, double the arrays, then gather and compare the sums.
"""
from mpi4py import MPI
import numpy as np

### TO RUN CODE, REMEMBER THE PREFIX mpiexec -n [num_processes] BEFORE python ./SimpleSummation ###

# MPI Initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


npoints = 100

if rank == 0:

    if npoints % size != 0:
        print("npoints must be divisible by number of processes")
        comm.Abort()

    # The full version of the array on the root process.
    # Each row will be sent to a different process, hence the reshape.
    full_array = np.linspace(1, 100, npoints, dtype=np.float64)
    full_array = full_array.reshape((size, npoints // size)) # // Floor divide as need int

    print(f"Initial sum: \n{np.sum(full_array)}")

else:
    # This variable will not get used on the non-root process, but
    # it needs to exist 
    full_array = None

    
# Initialising the destination array, root process also needs one
# np.zeros is actually faster than np.empty, weird I know
local_array = np.zeros(npoints // size, dtype=np.float64) # // Floor divide as need int

# Scatter the arrays across the processes - one row to each process
comm.Scatter(full_array, local_array, root=0)

# Doubling our array individually on each process
# Imagine this is a much more complicated piece of computation
local_array *= 2

# Gathering the arrays
comm.Gather(local_array, full_array, root=0)

if rank == 0:
    total_sum = np.sum(full_array)
    print(f"After MPI splitting and doubling, sum value is: \n{total_sum}")

# Job done, close out the MPI process
MPI.Finalize
