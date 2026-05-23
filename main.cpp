#include "cuda_kernel.cuh"

#include <cstdio>
#include <cstring>
#include <ctime>

#include <omp.h>

#ifndef CPU_BLOCK_SIZE
#define CPU_BLOCK_SIZE 64
#endif

int parse_non_negative(char* arg, int* ans) {
    char* endptr;
    int res = std::strtol(arg, &endptr, 10);
    if (*endptr != '\0' || res < 0) {
        std::fprintf(stderr, "should be non-negative decimal integer\n");
        return 1;
    }

    *ans = res;
    return 0;
}

int parseArgs(int argc, char* argv[], cmdArgs* args) {
    args->deviceIndex = 0;
    args->input = "input.txt";
    args->output = "output.txt";
    args->verify = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printf(
                "cuda prefsum\n"
                "--help \t\t- show help\n"
                "--verify \t- check cuda caluculations correctness with cpu\n"
                "--input \t- choose input file (default input.txt)\n"
                "--output \t- choose output file (default output.txt)\n"
                "--device-index \t- choose cuda device index\n"
            );
            std::exit(0);
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            args->verify = true;
        } else if (std::strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--input flag requires argument");
                return 1;
            }
            args->input = argv[i+1];
            i++;
        } else if (std::strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--output flag requires argument");
                return 1;
            }
            args->output = argv[i+1];
            i++;
        } else if (std::strcmp(argv[i], "--device-index") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--device-index flag requires argument");
                return 1;
            }

            if (parse_non_negative(argv[i+1], &args->deviceIndex)) {
                std::fprintf(stderr, "can't read device_index");
                return 1;
            }
        } else {
            std::fprintf(stderr, "no such flag \"%s\"", argv[i]);
            return 1;
        }
    }

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
    for (unsigned int b = 0; b < n; b += CPU_BLOCK_SIZE) {
        unsigned int brd = std::min((unsigned int) CPU_BLOCK_SIZE, n-b);
        res->arr[b] = task->arr[b];
        for (unsigned int i = 1; i < brd; i++) {
            res->arr[b+i] = res->arr[b+i-1] + task->arr[b+i];
        }
    }

    unsigned int ttl = 0;
    for (unsigned int i = CPU_BLOCK_SIZE-1; i < n; i += CPU_BLOCK_SIZE) {
        res->arr[i] += ttl;
        ttl = res->arr[i];
    }

    #pragma omp parallel for
    for (unsigned int b = CPU_BLOCK_SIZE; b < n; b += CPU_BLOCK_SIZE) {
        unsigned int brd = std::min((unsigned int) CPU_BLOCK_SIZE - 1, n-b);
        unsigned int pref = res->arr[b-1];
        for (unsigned int i = 0; i < brd; i++) {
            res->arr[b+i] += pref;
        }
    }

    end = clock();
    
    res->fullTime = ((double) (end - start) * 1000) / CLOCKS_PER_SEC;
        
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
        std::fprintf(stderr, "can't set cuda device with index %i: error %i", args.deviceIndex, cudaErr);
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
