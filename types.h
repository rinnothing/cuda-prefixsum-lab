#include <vector>

struct cmdArgs {
    int deviceIndex;
    char* input;
    char* output;
    bool verify;

    unsigned int cpuBlockSize;
};

struct calcTask {
    std::vector<unsigned int> arr;
};

struct calcRes {
    std::vector<unsigned int> arr;

    float kernelTime;
    float fullTime;
};
