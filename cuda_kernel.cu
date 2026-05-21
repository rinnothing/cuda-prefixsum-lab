#include "cuda_kernel.cuh"
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 8
#endif

__device__ void sklansky(unsigned int lx, unsigned int px, unsigned int gx, unsigned int* arr, unsigned int length) {
    //todo
}

enum status {
    A, P, X
};

struct blockDescriptor {
    volatile status status;
    unsigned int sum;
    unsigned int prefix;
};

__global__ void prefsum(blockDescriptor* descriptors, unsigned int descriptors_length, unsigned int* task, unsigned int task_length) {
    unsigned int lx = threadIdx.x;
    unsigned int px = blockDim.x * blockIdx.x;
    unsigned int gx = px + lx;

    __shared__ unsigned int local_task[BLOCK_SIZE];

    unsigned int end = std::min(task_length - px, (unsigned int) BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i += BLOCK_THREADS) {
        if (px + i < end) local_task[px + i] = task[px + i];
        else local_task[px+i] = 0;
    }

    __syncthreads();

    sklansky(lx, px, gx, local_task, task_length);

    __shared__ unsigned int block_prefix;

    if (lx == 0) {
        block_prefix = 0;
        descriptors[blockIdx.x].sum = task[BLOCK_SIZE-1];
        
        __threadfence();
        
        descriptors[blockIdx.x].status = status::P;

        for (unsigned int cur = blockIdx.x-1; cur >= 0; cur--) {
            while (descriptors[cur].status == status::X)

            if (descriptors[cur].status == status::P) {
                block_prefix += descriptors[cur].prefix;
                break;
            } else {
                block_prefix += descriptors[cur].sum;
            }
        }

        descriptors[blockIdx.x].prefix = block_prefix;
        
        __threadfence();

        descriptors[blockIdx.x].status = status::A;
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i += BLOCK_THREADS) {
        if (px + i < end) task[px + i] = block_prefix + local_task[px + i];
    }
}

int calcCUDA(cmdArgs* args, calcTask* task, calcRes* res) {
    unsigned int grid = task->arr.size() / BLOCK_SIZE;
    if (task->arr.size() % BLOCK_SIZE != 0) grid++;

    prefsum<<<grid, BLOCK_SIZE>>>();

    // todo
    return 0;
}
