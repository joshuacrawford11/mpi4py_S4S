from mpi4py import MPI
import numpy as np

### TO RUN CODE, REMEMBER THE PREFIX mpiexec -n [num_processes] BEFORE python ./SimpleSummation ###

# MPI Initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define array to sum on the head process
if rank == 0:
    npoints = 100
    if npoints % size != 0:
        print("npoints must be divisible by number of processes")
        comm.Abort()

    array_to_sum = np.linspace(1, 100, npoints)
    print(f"Expected value of sum: {np.sum(array_to_sum)} \n")

    # Split array into even chunks
    split_array = np.array_split(array_to_sum, size, axis=0)

else:
    # Define variable name on other processes to prevent errors
    split_array = None

# Scatter the arrays across the processes
# Yeah this isn't the "true" way to scatter an array but fight me
local_array = comm.bcast(split_array, root=0)[rank]

# Sum the array on the local process
local_sum = np.sum(local_array)

# Collect the sums, again, could be better, but refer above
total_sum = comm.gather(local_sum, root=0)

if rank == 0:
    total_sum = np.sum(total_sum)
    print(f"After MPI splitting, sum value is: {total_sum} \n")

# Job done, close out the MPI process
MPI.Finalize
