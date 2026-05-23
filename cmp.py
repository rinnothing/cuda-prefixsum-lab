import os
import time
import gc
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from gen import gen

BLOCK_SIZES = [256, 512, 1024, 2048, 4096]
BLOCK_THREADS = [4, 8, 16, 32]

EXECUTABLE = "./main.exe"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

def run_bench(block_size=1024, block_threads=8, cpu_block_size=None):
    my_env = os.environ.copy()
    my_env["PATH"] = f"{my_env["PATH"]};C:\\Program Files\\Microsoft Visual Studio 2022\\VC\\Tools\\MSVC\\14.42.34433\\bin\\Hostx64\\x64"

    compile_cmd = [
        "nvcc",
        "-o", "main.exe", "cuda_kernel.cu", "main.cpp",
        f"-D BLOCK_SIZES={block_size}",
        f"-D BLOCK_THREADS={block_threads}",
    ]

    if cpu_block_size is not None:
        compile_cmd.append(f"-D CPU_BLOCK_SIZE={cpu_block_size}")

    try:
        subprocess.run(compile_cmd, capture_output=True, text=True, check=True, env=my_env)
    except subprocess.CalledProcessError as e:
        if e.returncode != 2:
            print(f"Error compiling programm: {e}\n"
                f"output is:\n {e.stderr}")
            return None
    except Exception as e:
        print(f"Error compiling program: {e}\n")
        return None

    cmd = [
        EXECUTABLE,
        "--input", INPUT_FILE,
        "--output", OUTPUT_FILE,
    ]

    if cpu_block_size is not None:
        cmd.append("--verify")

    print(f"running option block_size={block_size}, block_threads={block_threads}, cpu_block_size={"None" if cpu_block_size is None else cpu_block_size}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        ans = None
        for line in result.stdout.split('\n'):
            if line.startswith("Time:"):
                parts = line.replace('\t', ' ').split()
                if len(parts) >= 3:
                    try:
                        ans = float(parts[2])
                    except ValueError:
                        continue
            if cpu_block_size is not None and line.startswith("Time CPU:"):
                parts = line.replace('\t', ' ').split()
                if len(parts) >= 3:
                    try:
                        ans = float(parts[2])
                    except ValueError:
                        continue
    except subprocess.CalledProcessError as e:
        if e.returncode != 2:
            print(f"Error running programm: {e}\n"
                f"output is:\n {e.stderr}")
            return None
    except Exception as e:
        print(f"Error running program: {e}\n")
        return None
    return ans

def main():
    results = {}

    print("Starting benchmarks...")
    for b_t in BLOCK_THREADS:
        for b_s in BLOCK_SIZES:
            t_ms = run_bench(b_s, b_t)
            results.setdefault(b_t, []).append(t_ms)

            time.sleep(0.1) 
            gc.collect()
    
    for c_b in BLOCK_SIZES:
        t_ms = run_bench(cpu_block_size=c_b)
        results.setdefault("cpu", []).append(t_ms)

        time.sleep(0.1) 
        gc.collect()

    if not os.path.exists("img"):
        os.makedirs("img")
        
    plt.figure(figsize=(10, 6))
    for key, times in results.items():
        plt.plot(BLOCK_SIZES, times, label=key)

    plt.title('Block size impact on different block thread number')
    plt.xlabel('Block size')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.savefig('img/comparison_with_cpu.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for key, times in results.items():
        if key == "cpu":
            continue
        plt.plot(BLOCK_SIZES, times, label=key)

    plt.title('Block size impact on different block thread number')
    plt.xlabel('Block size')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.savefig('img/comparison.png')
    plt.close()

    print("Graphs saved to img/ folder.")

if __name__ == "__main__":
    main()
