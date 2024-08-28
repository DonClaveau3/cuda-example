import numpy as np
from numba import cuda
import timeit

def increment_by_one_cpu(arr):
    for i in range(arr.size):
        arr[i] += 1

@cuda.jit
def increment_by_one_cuda(arr):
    pos = cuda.grid(1) # get absolute position in grid
    if pos < arr.size:  # Check array boundaries
        arr[pos] += 1    

def increment_by_one(arr,use_cuda=True):
    if use_cuda:
        threadsperblock = 32*2
        blockspergrid = (arr.size + (threadsperblock - 1)) // threadsperblock
        d_ary = cuda.to_device(arr)
        increment_by_one_cuda[blockspergrid, threadsperblock](d_ary)
        d_ary.copy_to_host(arr)
    else:
        increment_by_one_cpu(arr)
    
def RunTest(arrSize,use_cuda):
    arr = np.array([0]*arrSize)
    dur = timeit.timeit(lambda: increment_by_one(arr,use_cuda), number=5)    
    success = arr == np.array([5]*arrSize)
    print(arrSize,use_cuda,dur,success.all())
    return dur

for i in range(9):
    size = 10**i    
    dur_gpu = RunTest(size,True)
    dur_cpu = RunTest(size,False)
    if dur_cpu > dur_gpu:
        print("CUDA was",int(dur_cpu/dur_gpu),"times faster!")
    else:
        print("CUDA was",int(dur_gpu/dur_cpu),"times slower.")
