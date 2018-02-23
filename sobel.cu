#include "gpu.h"

__global__ void sobelKernel(double *dataIn,double *dataOut,int *out,int height,int width)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * width + xIndex;
    double Gx = 0;
    double Gy = 0;
    if (xIndex > 0 && xIndex < width - 1 && yIndex > 0 && yIndex < height - 1)
    {
        Gx = dataIn[(yIndex - 1) * width + xIndex + 1] + 2 * dataIn[yIndex * width + xIndex + 1] + dataIn[(yIndex + 1) * width + xIndex + 1]
            - (dataIn[(yIndex - 1) * width + xIndex - 1] + 2 * dataIn[yIndex * width + xIndex - 1] + dataIn[(yIndex + 1) * width + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * width + xIndex - 1] + 2 * dataIn[(yIndex - 1) * width + xIndex] + dataIn[(yIndex - 1) * width + xIndex + 1]
            - (dataIn[(yIndex + 1) * width + xIndex - 1] + 2 * dataIn[(yIndex + 1) * width + xIndex] + dataIn[(yIndex + 1) * width + xIndex + 1]);
        dataOut[index] = Gx*Gx+Gy*Gy;
		__syncthreads();
	}
	
	if(xIndex>1 && xIndex<width-2 && yIndex>1 && yIndex<height-2)
	{
		int flag=0;
		if(dataOut[index] <= dataOut[(yIndex - 1) * width + xIndex -1])
			flag +=0x80; 
		if(dataOut[index] <= dataOut[(yIndex - 1) * width + xIndex ])
			flag +=0x40;
		if(dataOut[index] <= dataOut[(yIndex - 1) * width + xIndex +1])
			flag +=0x20;
		if(dataOut[index] <= dataOut[(yIndex) * width + xIndex-1])
			flag +=0x10;
		if(dataOut[index] <= dataOut[(yIndex) * width + xIndex+1])
			flag +=0x08;
		if(dataOut[index] <= dataOut[(yIndex + 1) * width + xIndex-1])
			flag +=0x04;
		if(dataOut[index] <= dataOut[(yIndex + 1) * width + xIndex])
			flag +=0x02;
		if(dataOut[index] <= dataOut[(yIndex + 1) * width + xIndex+1])
			flag ++;
		out[index]=flag;
	}

}


void calculate_sobel_gpu(IntImage<REAL>& src,IntImage<int>& dst,int height,int width)
{
	//IntImage<REAL> sobel;
	//sobel.Create(height,width);
	dst.Create(height,width);
	cudaError_t cudaStatus;
	double *d_out;
	double *d_in;
	int *out;

	cudaStatus=cudaSetDevice(0);
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}
	//分配存储空间
	cudaStatus=cudaMalloc((void**)&d_in,height*width*sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus=cudaMalloc((void**)&d_out,height*width*sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus=cudaMalloc((void**)&out,height*width*sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//copy主机数据到设备
	cudaStatus=cudaMemcpy(d_in,src.buf,height*width*sizeof(double),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	dim3 threadsPerBlock(32,32);
	dim3 blocksPerGrid((width+threadsPerBlock.x-1)/threadsPerBlock.x,(height+threadsPerBlock.y-1)/threadsPerBlock.y);

	sobelKernel<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_out,out,height,width);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	cudaMemcpy(dst.buf, out, height * width * sizeof(int), cudaMemcpyDeviceToHost);//cudaMenmcpyToSymbol复制到常量内存
	
Error:
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(out);

}

__global__ void sobel_Kernel(double *dataIn,double *dataOut,int width,int height)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * width + xIndex;
    double Gx = 0;
    double Gy = 0;
    if (xIndex > 0 && xIndex < width - 1 && yIndex > 0 && yIndex < height - 1)
    {
        Gx = dataIn[(yIndex - 1) * width + xIndex + 1] + 2 * dataIn[yIndex * width + xIndex + 1] + dataIn[(yIndex + 1) * width + xIndex + 1]
            - (dataIn[(yIndex - 1) * width + xIndex - 1] + 2 * dataIn[yIndex * width + xIndex - 1] + dataIn[(yIndex + 1) * width + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * width + xIndex - 1] + 2 * dataIn[(yIndex - 1) * width + xIndex] + dataIn[(yIndex - 1) * width + xIndex + 1]
            - (dataIn[(yIndex + 1) * width + xIndex - 1] + 2 * dataIn[(yIndex + 1) * width + xIndex] + dataIn[(yIndex + 1) * width + xIndex + 1]);
        dataOut[index] = Gx*Gx+Gy*Gy;
	}
}
void calculate_sobel(IntImage<REAL>& image,IntImage<REAL>& src)
{
	src.Create(image.nrow,image.ncol);
	for(int i=0;i<image.nrow;i++) src.p[i][0] = src.p[i][image.ncol-1] = 0;
	std::fill(src.p[0],src.p[0]+image.ncol,0.0);
	std::fill(src.p[image.nrow-1],src.p[image.nrow-1],0.0);

	cudaError_t cudaStatus;
	double *d_out;
	double *d_in;
	int height=image.nrow;
	int width=image.ncol;
	cudaStatus=cudaSetDevice(0);
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}
	//分配存储空间
	cudaStatus=cudaMalloc((void**)&d_in,height*width*sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus=cudaMalloc((void**)&d_out,height*width*sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//copy主机数据到设备
	cudaStatus=cudaMemcpy(d_in,image.buf,height*width*sizeof(double),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	dim3 threadsPerBlock(32,32);
	dim3 blocksPerGrid((width+threadsPerBlock.x-1)/threadsPerBlock.x,(height+threadsPerBlock.y-1)/threadsPerBlock.y);

	sobel_Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_out,width,height);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	cudaMemcpy(src.buf, d_out, height * width * sizeof(double), cudaMemcpyDeviceToHost);//cudaMenmcpyToSymbol复制到常量内存

Error:
	cudaFree(d_in);
	cudaFree(d_out);

}

__global__ void ct_Kernel(double *dataIn,int *dataOut,int width,int height)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * width + xIndex;
	if(xIndex>1 && xIndex<width-2 && yIndex>1 && yIndex<height-2)
	{
		int flag=0;
		if(dataIn[index] <= dataIn[(yIndex - 1) * width + xIndex -1])
			flag +=0x80; 
		if(dataIn[index] <= dataIn[(yIndex - 1) * width + xIndex ])
			flag +=0x40;
		if(dataIn[index] <= dataIn[(yIndex - 1) * width + xIndex +1])
			flag +=0x20;
		if(dataIn[index] <= dataIn[(yIndex) * width + xIndex-1])
			flag +=0x10;
		if(dataIn[index] <= dataIn[(yIndex) * width + xIndex+1])
			flag +=0x08;
		if(dataIn[index] <= dataIn[(yIndex + 1) * width + xIndex-1])
			flag +=0x04;
		if(dataIn[index] <= dataIn[(yIndex + 1) * width + xIndex])
			flag +=0x02;
		if(dataIn[index] <= dataIn[(yIndex + 1) * width + xIndex+1])
			flag ++;
		dataOut[index]=flag;
	}
}
void calculate_ct(IntImage<REAL>& src,IntImage<int>& dst)
{
	dst.Create(src.nrow,src.ncol);
	cudaError_t cudaStatus;

	double *d_in;
	int *d_out;
	int height=src.nrow;
	int width=src.ncol;
	cudaStatus=cudaSetDevice(0);
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}
	//分配存储空间
	cudaStatus=cudaMalloc((void**)&d_in,height*width*sizeof(double));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus=cudaMalloc((void**)&d_out,height*width*sizeof(int));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//copy主机数据到设备
	cudaStatus=cudaMemcpy(d_in,src.buf,height*width*sizeof(double),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	dim3 threadsPerBlock(32,32);
	dim3 blocksPerGrid((width+threadsPerBlock.x-1)/threadsPerBlock.x,(height+threadsPerBlock.y-1)/threadsPerBlock.y);

	ct_Kernel<<<blocksPerGrid,threadsPerBlock>>>(d_in,d_out,width,height);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
	cudaMemcpy(dst.buf, d_out, height * width * sizeof(int), cudaMemcpyDeviceToHost);//cudaMenmcpyToSymbol复制到常量内存

Error:
	cudaFree(d_in);
	cudaFree(d_out);
}