#include <vector>

struct cmdArgs {
    int deviceIndex;
    char* input;
    char* output;
    bool verify;
};

struct calcTask {
    std::vector<unsigned int> arr;
};

struct calcRes {
    std::vector<unsigned int> arr;

    float kernelTime;
    float fullTime;
};
