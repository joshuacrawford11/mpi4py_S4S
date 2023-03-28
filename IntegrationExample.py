from mpi4py import MPI

# MPI initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define parameters for integration
if rank == 0:
    # Integrate on (a,b) with n trapezoids
    a = 0
    b = 1
    n = 10000
    # Predefine total value
    total = -1
else:
    a = None
    b = None
    n = None
    dest = 0
    total = -1.0

# Define a function to integrate
def FuncToIntegrate(x):
    # Can modify at a later date, but let's just consider x^2 for now.
    f = x**2
    return f

# Trapezoid routine for computing an integral numerically
def Trapezoid(a,b,n,h):

    integral = (FuncToIntegrate(a) + FuncToIntegrate(b))/2.0

    x=a

    for ii in range(1,int(n)):
        x = x + h

        integral += FuncToIntegrate(x)

    return integral*h

# Broadcast integral parameters to all processes
a = comm.bcast(a)
b = comm.bcast(b)
n = comm.bcast(n)

h = (b-a)/n
local_n = n/size

# Compute the local integral
local_a = a + rank*local_n*h
local_b = local_a + local_n*h
integral = Trapezoid(local_a, local_b, local_n, h)

# Reduce the result to the root process
total = comm.reduce(integral)

# Print the result
if (rank ==0):
    print("With n = ", n, " trapezoids, \n")
    print("integral from ", a, " to ", b, " = ", total, "\n")

# Finalise process
MPI.Finalize
