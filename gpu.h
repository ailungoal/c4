#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "IntImage.h"
#include "device_functions.h"
#include <opencv2\opencv.hpp>

void calculate_sobel_gpu(IntImage<REAL>& src,IntImage<int>& result,int height,int width);

void calculate_sobel(IntImage<REAL>& image,IntImage<REAL>& src);

void calculate_integral(IntImage<double>& src,int xoffset,int yoffset);

void calculate_ct(IntImage<REAL>& src,IntImage<int>& dst);