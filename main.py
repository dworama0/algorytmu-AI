from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer

def cpu_function(a):
    for i in range(100_000_000):
        a[i]+=1

@jit(nopython=True)
def gpu_function(a):
    for i in range(100_000_000):
        a[i]+=1

if __name__ == '__main__':
    n:float = 100_000_000
    a = np.ones(n,dtype=np.float64)

    start = timer()
    cpu_function(a)
    print(f'funkcja CPU wykonana w  {timer() - start}s')
    print("_"*50)
    start = timer()
    gpu_function(a)
    print(f'funkcja GPU wykonana w  {timer() - start}s')