#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "helper.h"

void checkErrorCuda(cudaError_t cudaStatus)
{
    if (cudaStatus != cudaSuccess)
        {
            printf("Cuda Error: %s\n", cudaGetErrorString(cudaStatus));

            exit(1);
        }
}

void checkErrorMemory(void* val)
{
    if (val == NULL)
        {
            printf("Error allocating memory in CudaHelper\n");
            exit(1);
        }
}


__global__ void calculateMatching(int *d_pictureMat, int* pictureDim, int *d_objectMat, int* objectDim, double* d_matchingThreshold, int* d_indexFound, int* d_flag)
{
    int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = ((*pictureDim) - (*objectDim) + 1) * ((*pictureDim) - (*objectDim) + 1) ;

    if (globalThreadIndex < limit) {
        int rowIndex = globalThreadIndex / ((*pictureDim) - (*objectDim) + 1);    
        int colIndex = globalThreadIndex % ((*pictureDim) - (*objectDim) + 1);   
        int offset = rowIndex * (*pictureDim) + colIndex;   
        double match = 0;
        
        for (int i = 0; i < (*objectDim); i++){
            for(int  j = 0; j < (*objectDim); j++){
                int pictureVal = *(d_pictureMat + offset + i*(*pictureDim) + j);   
                int objectVal = *(d_objectMat + i*(*objectDim) + j);
                match += abs((pictureVal - objectVal) / pictureVal);
            }
        }

        if (match / ((*objectDim) * (*objectDim)) <  *d_matchingThreshold)
        {
            *d_flag = 1;
            *(d_indexFound + globalThreadIndex * 2 ) = rowIndex;
            *(d_indexFound + globalThreadIndex * 2 + 1) = colIndex;
        }
    }
}

__host__ void calculateMatchingOnGPU(Picture *picture, Object *object, double *matchingThreshold, int* foundFlag, int* indexsFound, int size)
{
    int* d_flag;
    checkErrorCuda(cudaMalloc((void **)&d_flag, sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_flag, foundFlag, sizeof(int), cudaMemcpyHostToDevice));

    // allocate memory for indexs match array 
    int* d_indexsFound;
    checkErrorCuda(cudaMalloc((void**)&d_indexsFound, 2 * size * sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_indexsFound, indexsFound, 2 * size * sizeof(int), cudaMemcpyHostToDevice));


    // allocate memory for the result on the GPU
    double* d_matchingThreshold;
    checkErrorCuda(cudaMalloc((void **)&d_matchingThreshold, sizeof(double)));
    checkErrorCuda(cudaMemcpy(d_matchingThreshold, matchingThreshold, sizeof(double), cudaMemcpyHostToDevice));

    // allocate memory for the picture dimension on the GPU and copy the picture dimension to the GPU
    int* d_pictureDim;
    checkErrorCuda(cudaMalloc((void **)&d_pictureDim, sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_pictureDim, &picture->dim, sizeof(int), cudaMemcpyHostToDevice));

    // allocate memory for the object dimension on the GPU and copy the object dimension to the GPU
    int* d_objectDim;
    checkErrorCuda(cudaMalloc((void **)&d_objectDim, sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_objectDim, &object->dim, sizeof(int), cudaMemcpyHostToDevice));

    // allocate memory for the picture colors matrix on the GPU and copy the picture colors matrix to the GPU
    int* d_pictureMat;
    checkErrorCuda(cudaMalloc((void **)&d_pictureMat, picture->dim * picture->dim * sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_pictureMat, picture->mat, picture->dim * picture->dim * sizeof(int), cudaMemcpyHostToDevice));

    // allocate memory for the object sub colors matrix on the GPU and copy the object sub colors matrix to the GPU
    int* d_objectMat;
    checkErrorCuda(cudaMalloc((void **)&d_objectMat, object->dim * object->dim * sizeof(int)));
    checkErrorCuda(cudaMemcpy(d_objectMat, object->mat, object->dim * object->dim * sizeof(int), cudaMemcpyHostToDevice));


    int blockSize = 512;                                    // threads per block
    int numBlocks = (size + blockSize - 1) / blockSize;     // block per grid
    
    // call the kernel function 
    calculateMatching <<<numBlocks, blockSize >>> (d_pictureMat, d_pictureDim, d_objectMat, d_objectDim, d_matchingThreshold, d_indexsFound, d_flag);

    checkErrorCuda(cudaDeviceSynchronize());

    // check if the kernel function was called successfully
    checkErrorCuda(cudaGetLastError());

    // copy the matched indexs to CPU
    checkErrorCuda(cudaMemcpy(indexsFound, d_indexsFound, 2 * size * sizeof(int), cudaMemcpyDeviceToHost));

    // copy found object flag
    checkErrorCuda(cudaMemcpy(foundFlag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));    
    if(*foundFlag == 1)
    {
        (*picture).objFound++;
    }
    // free the memory on the GPU
    cudaFree(d_matchingThreshold);
    cudaFree(d_pictureMat);
    cudaFree(d_objectMat);
    cudaFree(d_pictureDim);
    cudaFree(d_objectDim);
    cudaFree(d_indexsFound);

}

