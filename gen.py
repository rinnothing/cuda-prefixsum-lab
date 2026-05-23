from random import random
import struct

def binary_uint32_t(num):
    return struct.pack("I", num)

def gen_arr(x):
    arr = []
    for i in range(x):
        arr.append(int(random() * 10))

    return arr


def write_arr(f, arr):
    for i in range(len(arr)):
        f.write(binary_uint32_t(arr[i]))
            # f.write(f'{arr[j][i]:.2f} ')
        # f.write('\n')

def gen(name, x):
    a = gen_arr(x)

    f = open(name, "wb")
    # f.write(f'{x} {m} {y}\n')
    f.write(binary_uint32_t(x))

    write_arr(f, a)

if __name__ == "__main__":
    x = 1600000

    gen("input.txt", x)
