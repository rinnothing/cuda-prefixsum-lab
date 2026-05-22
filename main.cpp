#include "cuda_kernel.cuh"

#include <cstdio>
#include <ctime>

#include <omp.h>

int parseArgs(int argc, char* argv[], cmdArgs* args) {
    args->deviceIndex = 0;
    args->input = "input.txt";
    args->output = "output.txt";
    args->verify = false;

    args->cpuBlockSize = 64;

    
    // debug
    // args->verify = true;
    args->cpuBlockSize = 16;

    //todo
    return 0;
}

int readTask(cmdArgs* args, calcTask* task) {
    std::FILE* f = std::fopen(args->input, "rb");
    if (f == NULL) {
        std::fprintf(stderr, "can't open input file");
        return 1;
    }

    unsigned int n;
    if (std::fread(&n, sizeof(unsigned int), 1, f) != 1) {
        std::fprintf(stderr, "can't parse task length");
        std::fclose(f);
        return 1;
    }

    task->arr.resize(n);
    if (std::fread(&(*task->arr.begin()), sizeof(unsigned int), n, f) != n) {
        std::fprintf(stderr, "can't parse task contents");
        std::fclose(f);
        return 1;
    }

    
    //debug
    // std::printf("task values are:");
    // for (int i = 0; i < n; i++) {
    //     std::printf(" %i", task->arr[i]);
    // }
    // std::printf("\n");

    return 0;
}

int writeRes(cmdArgs* args, calcRes* res) {
    std::FILE* f = std::fopen(args->output, "wb");
    if (f == NULL) {
        std::fprintf(stderr, "can't open output file");
        return 1;
    }

    unsigned int n = res->arr.size();
    if (std::fwrite(&n, sizeof(unsigned int), 1, f) != 1) {
        std::fprintf(stderr, "can't write res length");
        std::fclose(f);
        return 1;
    }

    if (std::fwrite(&(*res->arr.begin()), sizeof(unsigned int), n, f) != n) {
        std::fprintf(stderr, "can't write res contents");
        std::fclose(f);
        return 1;
    }

    return 0;
}

int calcCPU(cmdArgs* args, calcTask* task, calcRes* res) {
    unsigned int n = task->arr.size();
    res->arr.resize(n);

    std::clock_t start, end;
    start = clock();

    #pragma omp parallel for
    for (unsigned int b = 0; b < n; b += args->cpuBlockSize) {
        unsigned int brd = std::min(args->cpuBlockSize, n-b);
        res->arr[b] = task->arr[b];
        for (unsigned int i = 1; i < brd; i++) {
            res->arr[b+i] = res->arr[b+i-1] + task->arr[b+i];
        }
    }

    unsigned int ttl = 0;
    for (unsigned int i = args->cpuBlockSize-1; i < n; i += args->cpuBlockSize) {
        res->arr[i] += ttl;
        ttl = res->arr[i];
    }

    #pragma omp parallel for
    for (unsigned int b = args->cpuBlockSize; b < n; b += args->cpuBlockSize) {
        unsigned int brd = std::min(args->cpuBlockSize - 1, n-b);
        unsigned int pref = res->arr[b-1];
        for (unsigned int i = 0; i < brd; i++) {
            res->arr[b+i] += pref;
        }
    }

    end = clock();
    
    res->fullTime = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;

    //debug
    // std::printf("cpu values are:");
    // for (int i = 0; i < n; i++) {
    //     std::printf(" %i", res->arr[i]);
    // }
    // std::printf("\n");
    
    return 0;
}

int compareResults(cmdArgs* args, calcRes* cudaRes, calcRes* cpuRes) {
    if (cudaRes->arr.size() != cpuRes->arr.size()) {
        std::fprintf(stderr, "results length aren't equal cpu - %zi, cuda - %zi", cpuRes->arr.size(), cudaRes->arr.size());
        return 1;
    }
    
    int n = cudaRes->arr.size();
    for (int i = 0; i < n; i++) {
        if (cudaRes->arr[i] != cpuRes->arr[i]) {
            std::fprintf(stderr, "results are different in %i symbol - cpu %i, cuda %i", i, cpuRes->arr[i], cudaRes->arr[i]);
            return 1;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    cmdArgs args;
    int err = parseArgs(argc, argv, &args);
    if (err) {
        return err;
    }

    int devCount;
    cudaError_t cudaErr = cudaGetDeviceCount(&devCount);
    if (cudaErr) {
        std::fprintf(stderr, "can't get cuda device number: error %i", cudaErr);
        return 1;
    }

    args.deviceIndex %= devCount;

    cudaErr = cudaSetDevice(args.deviceIndex);
    if (cudaErr) {
        std::fprintf(stderr, "can't set cuda device: error %i", cudaErr);
        return 1;
    }

    cudaDeviceProp devProp;
    cudaErr = cudaGetDeviceProperties(&devProp, args.deviceIndex);
    if (cudaErr) {
        std::fprintf(stderr, "can't get cuda device properties: error %i", cudaErr);
        return 1;
    }

    int driverVersion;
    cudaErr = cudaDriverGetVersion(&driverVersion);
    if (cudaErr) {
        std::fprintf(stderr, "can't get cuda driver version: error %i", cudaErr);
        return 1;
    }

    std::printf("Device: %s\tDriver version: %i\n", devProp.name, driverVersion);

    calcTask task;
    err = readTask(&args, &task);
    if (err) {
        return err;
    }

    calcRes cudaRes;
    err = calcCUDA(&args, &task, &cudaRes);
    if (err) {
        return err;
    }

    cudaErr = cudaDeviceReset();
    if (cudaErr) {
        std::fprintf(stderr, "can't reset cuda device: error %i", cudaErr);
        return 1;
    }

    std::printf("Time: %g\t%g\n", cudaRes.kernelTime, cudaRes.fullTime);

    err = writeRes(&args, &cudaRes);
    if (err) {
        return err;
    }

    if (args.verify) {
        calcRes cpuRes;
        err = calcCPU(&args, &task, &cpuRes);
        if (err) {
            return err;
        }

        std::printf("Time CPU: %g\n", cpuRes.fullTime);

        err = compareResults(&args, &cudaRes, &cpuRes);
        if (err) {
            return err;
        }
    }

    return 0;
}
