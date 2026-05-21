#include "cuda_kernel.h"

#include <cstdio>
#include <vector>

struct cmdArgs {
    int deviceIndex;
    char* input;
    char* output;
    bool verify;
};

int parseArgs(int argc, char* argv[], cmdArgs* args) {
    args->deviceIndex = 0;
    args->input = "input.txt";
    args->output = "output.txt";
    args->verify = false;

    //todo
    return 0;
}

struct calcTask {
    std::vector<unsigned int> arr;
};

int readTask(cmdArgs* args, calcTask* task) {
    // todo
    return 0;
}

struct calcRes {
    std::vector<unsigned int> arr;

    float kernelTime;
    float fullTime;
};

int writeRes(cmdArgs* args, calcRes* res) {
    // todo
    return 0;
}

int calcCUDA(cmdArgs* args, calcTask* task, calcRes* res) {
    // todo
    return 0;
}

int calcCPU(calcTask* task, calcRes* res) {
    // todo
    return 0;
}

int compareResults(calcRes* cudaRes, calcRes* cpuRes) {
    // todo
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
        fprintf(stderr, "can't get cuda device number: error %i", cudaErr);
        return 1;
    }

    args.deviceIndex %= devCount;

    cudaDeviceProp devProp;
    cudaErr = cudaGetDeviceProperties(&devProp, args.deviceIndex);
    if (cudaErr) {
        fprintf(stderr, "can't get cuda device properties: error %i", cudaErr);
        return 1;
    }

    int driverVersion;
    cudaErr = cudaDriverGetVersion(&driverVersion);
    if (cudaErr) {
        fprintf(stderr, "can't get cuda driver version: error %i", cudaErr);
    }

    printf("Device: %s\tDriver version: %i\n", devProp.name, driverVersion);

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

    printf("Time: %g\t%g\n", cudaRes.kernelTime, cudaRes.fullTime);

    err = writeRes(&args, &cudaRes);
    if (err) {
        return err;
    }

    if (args.verify) {
        calcRes cpuRes;
        err = calcCPU(&task, &cpuRes);
        if (err) {
            return err;
        }

        printf("Time CPU: %g\n", cpuRes.fullTime);

        err = compareResults(&cudaRes, &cpuRes);
        if (err) {
            return err;
        }
    }

    return 0;
}
