# Fast 3x3 SVDs for GPUs and CPUs
This project is a CUDA and C++ based implementation of the technical report "Computing the Singular Value Decomposition of 3x3 matrices with minimal branching and elementary floating point operations" available here: http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html

The original authors of the report offer a very optimized implementation for CPUs which is difficult to understand and use and is based on very basic C style programming. This project offers a simple 3x3 Matrix class and a simple function to call.

This project can be used by simply including the SVD.h file which works for GPU and CPU code. If only CPU execution is supported then `NO_CUDA_SUPPORT` should be defined in SVD.h, or before including the header.

# How to use
All that is required to test this result is to use the following code:
```cpp
// Create a matrix
SVD::Mat3x3 m = randomMatrix();
// Call the actual SVD decomposition which returns a struct containing the 3 calculated matrices U, S, V
auto [U, S, V] = SVD::svd(m);
// Reconstruct B from the SVD
auto B = U * S * V.transpose();
// Simple error metric
auto error = (A - B).det();
```
In order to use a user defined matrix the SVD::Mat3x3 class provides a constructor taking the elements of the matrix in row major order.

# Testing
The supplied kernel.cu function can be used to test the SVD method by generating 16M random 3x3 matrices with norm 1 and calculating the error of the program execution. An example output could look like this:
```
bin> .\testSVD.exe
Generating random test data
 16777200/16777216 [=======================================================================================================================>]  99%     2361438.01 elems/sec  Elapsed:  0h  0m  7s  104ms ETA:  0h  0m  0s    0ms
Running test on CPU
 16777216/16777216 [========================================================================================================================] 100%      159147.74 elems/sec  Elapsed:  0h  1m 45s  419ms ETA:  0h  0m  0s    0ms
Running test on GPU
 16777216/16777216 [========================================================================================================================] 100%  1243954622.97 elems/sec  Elapsed:  0h  0m  0s   13ms ETA:  0h  0m  0s    0ms
Absolute Errors:
Data entries: 16777216, avg:3.282606e-12 [0x1.cdfc58p-39], median: 8.666991e-13 [0x1.e7e880p-41], min: 0.000000e+00 [0x0.000000p+0], max: 3.232229e-10 [0x1.636326p-32], stddev: 7.473905e-12 [0x1.06f6f4p-37]
Data entries: 16777216, avg:3.282606e-12 [0x1.cdfc58p-39], median: 8.666991e-13 [0x1.e7e880p-41], min: 0.000000e+00 [0x0.000000p+0], max: 3.232229e-10 [0x1.636326p-32], stddev: 7.473905e-12 [0x1.06f6f4p-37]
9645154  x
9002144  x
8359133  x
7716123  x
7073113  x
6430102  x
5787092  x
5144082  x
4501072  x
3858061  x
3215051  x
2572041  x
1929030  xx
1286020  xx
643010   xxxx
0        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx x xxxxxxxx  xx x x xxx x  xxxx xx      x x              x x

Maximum Errors:
Data entries: 16777216, avg:1.255601e-12 [0x1.616b8ep-40], median: 2.877698e-13 [0x1.440000p-42], min: 0.000000e+00 [0x0.000000p+0], max: 1.464038e-10 [0x1.41f200p-33], stddev: 2.919562e-12 [0x1.9ae44cp-39]
Data entries: 16777216, avg:1.255601e-12 [0x1.616b8ep-40], median: 2.877698e-13 [0x1.440000p-42], min: 0.000000e+00 [0x0.000000p+0], max: 1.464038e-10 [0x1.41f200p-33], stddev: 2.919562e-12 [0x1.9ae44cp-39]
10393088 x
9700215  x
9007342  x
8314470  x
7621597  x
6928725  x
6235852  x
5542980  x
4850107  x
4157235  x
3464362  x
2771490  x
2078617  x
1385745  xx
692872   xxx
0        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xx xx xxxxxxxx xx x x xxx x  x x xxx  x  x            x    x                          xx

Diagonal Errors:
Data entries: 16777216, avg:1.402221e-06 [0x1.7867e0p-20], median: 1.246235e-06 [0x1.4e88a0p-20], min: 8.350538e-09 [0x1.1eec1cp-27], max: 1.056105e-05 [0x1.625eb4p-17], stddev: 1.081538e-06 [0x1.2252c0p-20]
Data entries: 16777216, avg:1.402221e-06 [0x1.7867e0p-20], median: 1.246235e-06 [0x1.4e88a0p-20], min: 8.350538e-09 [0x1.1eec1cp-27], max: 1.056105e-05 [0x1.625eb4p-17], stddev: 1.081538e-06 [0x1.2252c0p-20]
585825       xx
546770      xxxx
507715      xxxxx
468660      xxxxx
429605      xxxxxx
390550      xxxxxxx
351495     xxxxxxxxx
312440     xxxxxxxxxx         xxxxxxxx
273385     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
234330     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
195275     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
156220     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
117165    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
78110     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
39055     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
0        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxx  x  x x x         x
```
The errors shown are the absolute (sum of square) error, the worst absolute error on a single element and the magnitude of the non diagonal elements in the S matrix. The first line corresponds to the CPU results and the second to the GPU results.

# Numerical Accuracy
Care has been taken to make sure that the results of both GPU and CPU execution are exactly identical to avoid strange results due to compiler optimizations inserting FMAD operations that can change the numerical accuracy. As such both the GPU and CPU version use FTZ behaviour and the code includes manually inserted fmaf instructions to guarantee exactly the same results. The implementation uses a manually defined rsqrt function to guarantee the same precision on GPU and CPU results.

The accuracy of the results itself can be controlled by changing the macros in the SVD.h file:
```cpp
#define JACOBI_STEPS 12
#define RSQRT_STEPS 4
#define RSQRT1_STEPS 6
```
Which control the number of jacobi steps used to calculate the quaternion (see the referenced technical report) and the number of newton steps used in the provided rsqrt functions. 

