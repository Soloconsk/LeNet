import numpy as np


def InitKernel_C1():
    a = np.random.normal(size=(6, 5, 5))
    return a


if __name__ == '__main__':
    b = InitKernel_C1()
    print(b, b.shape)
