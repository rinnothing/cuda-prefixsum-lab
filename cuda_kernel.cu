#include "cuda_kernel.cuh"
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 8
#endif

__device__ void sklansky(unsigned int lx, unsigned int px, unsigned int* arr) {
    unsigned int pref = 0;
    for (unsigned int base = px; base < px + BLOCK_SIZE; px += BLOCK_THREADS * 2) {
        int i = lx * 2 + 1;

        int step_deg = 1;
        for (int step = 1; step <= BLOCK_THREADS; step <<= 1) {
            arr[base + i] += arr[base + i - step];

            __syncthreads();

            if (lx - ((lx >> step_deg) << step_deg) < step) i += step;

            step_deg++;
        }

        for (int j = 0; j < 2; j++) {
            arr[base + lx + j * BLOCK_SIZE] += pref;
        }

        __syncthreads();

        pref = arr[px + BLOCK_THREADS * 2 - 1];
    }
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

    __shared__ unsigned int local_task[BLOCK_SIZE];

    unsigned int end = min(task_length - px, (unsigned int) BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i += BLOCK_THREADS) {
        if (px + i < end) local_task[px + i] = task[px + i];
        else local_task[px+i] = 0;
    }

    __syncthreads();

    sklansky(lx, px, local_task);

    __shared__ unsigned int block_prefix;

    if (lx == 0) {
        block_prefix = 0;
        descriptors[blockIdx.x].sum = task[BLOCK_SIZE-1];
        
        __threadfence();
        
        descriptors[blockIdx.x].status = status::P;

        for (int cur = blockIdx.x-1; cur >= 0; cur--) {
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

struct cudaContext {
    cudaEvent_t events[4];
    blockDescriptor* descriptors;
    unsigned int* tasks;
};

cudaContext createCudaContext() {
    cudaContext ctx = {{NULL, NULL, NULL, NULL}, NULL, NULL};

    return ctx;
}

void cleanCudaContext(cudaContext* ctx) {
    cudaError_t cudaErr = cudaError::cudaSuccess;
    if (ctx->descriptors) {
        cudaErr = cudaFree(ctx->descriptors);
        if (cudaErr) {
            std::fprintf(stderr, "can't free descriptors block memory on cuda: error %i\n", cudaErr);
        }
    }
    if (ctx->tasks) {
        cudaErr = cudaFree(ctx->tasks);
        if (cudaErr) {            
            std::fprintf(stderr, "can't free tasks memory on cuda: error %i\n", cudaErr);
        }
    }
    for (int i = 0; i < 4; i++) {
        if (ctx->events[i]) {
            cudaErr = cudaEventDestroy(ctx->events[i]);
            if (cudaErr) {
                std::fprintf(stderr, "can't destroy cuda event %i: error %i\n", i, cudaErr);
            }
        }
    }
}

int recordAndSynchronizeEvent(cudaEvent_t event) {
    cudaError_t cudaErr = cudaEventRecord(event);
    if (cudaErr) {
        std::fprintf(stderr, "can't record event: error %i\n", cudaErr);
        return 1;
    }

    cudaErr = cudaEventSynchronize(event);
    if (cudaErr) {
        std::fprintf(stderr, "can't synchronize event: error %i\n", cudaErr);
        return 1;
    }

    return 0;
}

int calcCUDA(cmdArgs* args, calcTask* task, calcRes* res) {
    unsigned int grid = task->arr.size() / BLOCK_SIZE;
    if (task->arr.size() % BLOCK_SIZE != 0) grid++;

    std::vector<blockDescriptor> descriptors_device(grid);
    for (int i = 0; i < grid; i++) {
        descriptors_device[i].status = status::X;
    }

    cudaError_t cudaErr = cudaError::cudaSuccess;
    cudaContext ctx = createCudaContext();
    for (int i = 0; i < 4; i++) {
        cudaErr = cudaEventCreate(&ctx.events[i]);
        if (cudaErr) {
            std::fprintf(stderr, "can't create cuda event %i: error %i\n", i, cudaErr);
            cleanCudaContext(&ctx);
            return 1;
        }
    }

    int err = recordAndSynchronizeEvent(ctx.events[0]);
    if (err) {
        cleanCudaContext(&ctx);
        return err;
    }

    cudaErr = cudaMalloc(&ctx.descriptors, sizeof(blockDescriptor) * grid);
    if (cudaErr) {
        std::fprintf(stderr, "can't allocate cuda descriptors: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    cudaErr = cudaMemcpyAsync(ctx.descriptors, &(*descriptors_device.begin()), sizeof(blockDescriptor) * grid, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (cudaErr) {
        std::fprintf(stderr, "can't copy cuda descriptors from host to device: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    cudaErr = cudaMalloc(&ctx.tasks, sizeof(unsigned int) * task->arr.size());
    if (cudaErr) {
        std::fprintf(stderr, "can't allocate cuda tasks: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    cudaErr = cudaMemcpyAsync(ctx.tasks, &(*task->arr.begin()), sizeof(unsigned int) * task->arr.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (cudaErr) {
        std::fprintf(stderr, "can't copy cuda task from host to device: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    err = recordAndSynchronizeEvent(ctx.events[1]);
    if (err) {
        cleanCudaContext(&ctx);
        return err;
    }

    prefsum<<<grid, BLOCK_SIZE>>>(ctx.descriptors, grid, ctx.tasks, task->arr.size());

    err = recordAndSynchronizeEvent(ctx.events[2]);
    if (err) {
        cleanCudaContext(&ctx);
        return err;
    }

    res->arr.resize(task->arr.size());
    cudaErr = cudaMemcpyAsync(&(*res->arr.begin()), ctx.tasks, sizeof(unsigned int) * res->arr.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if (cudaErr) {
        std::fprintf(stderr, "can't copy cuda result from device to host: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    err = recordAndSynchronizeEvent(ctx.events[3]);
    if (err) {
        cleanCudaContext(&ctx);
        return err;
    }

    cudaErr = cudaEventElapsedTime(&res->fullTime, ctx.events[0], ctx.events[3]);
    if (cudaErr) {
        std::fprintf(stderr, "can't calculate full kernel time: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    cudaErr = cudaEventElapsedTime(&res->kernelTime, ctx.events[1], ctx.events[2]);
    if (cudaErr) {
        std::fprintf(stderr, "can't calculate calculation kernel time: error %i\n", cudaErr);
        cleanCudaContext(&ctx);
        return 1;
    }

    cleanCudaContext(&ctx);
    return 0;
}
