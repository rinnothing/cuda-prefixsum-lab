#!/bin/bash

PATH="$PATH:/c/Program Files/Microsoft Visual Studio 2022/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/" nvcc -o main.exe cuda_kernel.cu main.cpp -D WORD=\"bebra\"
