# cuda-example
short parallel program demonstrating use of cuda to task gpu in a python script

# Usage  
```
git clone git@github.com:DonClaveau3/cuda-example.git
cd cuda-example
python ./demo.py
```

# Test conditions
tested in Windows Subsystem for Linux environment on personal laptop    
```
WSL version: 2.2.4.0  
Kernel version: 5.15.153.1-2  
Windows version: 10.0.22631.4037  
Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 2001 Mhz, 4 Core(s), 8 Logical Processor(s)  
NVIDIA GeForce GTX 1050 (Driver Version	32.0.15.6094)
cuda-12-6/unknown 12.6.0-1 amd64
Python 3.10.12
Numba 0.60.0
Numpy 2.0.2
```

# Test results
```
home/donclaveau3/.pyenv/versions/3.10.12/lib/python3.10/site-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
1 True 0.857207228000334 True
1 False 1.5300000086426735e-05 True
CUDA was 56026 times slower.
10 True 0.00511330599692883 True
10 False 3.600000127335079e-05 True
CUDA was 142 times slower.
/home/donclaveau3/.pyenv/versions/3.10.12/lib/python3.10/site-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
100 True 0.006730008000886301 True
100 False 0.00014370000280905515 True
CUDA was 46 times slower.
/home/donclaveau3/.pyenv/versions/3.10.12/lib/python3.10/site-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
1000 True 0.010784912999952212 True
1000 False 0.004496206001931569 True
CUDA was 2 times slower.
10000 True 0.013352916001167614 True
10000 False 0.01825442199697136 True
CUDA was 1 times faster!
100000 True 0.013457117001962615 True
100000 False 0.12071444499815698 True
CUDA was 8 times faster!
1000000 True 0.039060947001416935 True
1000000 False 1.141806870000437 True
CUDA was 29 times faster!
10000000 True 0.28458124099779525 True
10000000 False 10.609542010999576 True
CUDA was 37 times faster!
100000000 True 2.9802367909978784 True
100000000 False 102.95512928000244 True
CUDA was 34 times faster!
```

# Helpful Resources 
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://numba.readthedocs.io/en/stable/cuda/index.html
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
