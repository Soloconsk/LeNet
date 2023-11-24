import numpy as np


def C1(input, kernel, inchannel, outchannel):
    sum_v = 0  # S2中unit的值
    i_count = 0
    j_count = 0  # 被C1卷积出来的值放入S2的位置
    output = np.zeros((28, 28, 6))
    for index in range(0, outchannel):  # S2中feature map的下标
        for i in range(0, 32):
            for j in range(0, 32):  # 卷积一次
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j] * kernel[kernel_i][kernel_j]

                output[i_count][j_count][index] = sum_v
                j_count += 1  # C1中的5x5被卷积计算完毕了，可以更新到下一个卷积块了
        i_count += 1  # C1中的一行被卷积计算完毕了，可以更新到下一行了
    return output


def S2(input, kernel, inchannel, outchannel):
    sum_v = 0
    i_count = 0
    j_count = 0
    output = np.zeros((14, 14, 6))
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            for kernel_i in range(i, i + 2):
                for kernel_j in range(j, j + 2):
                    input[kernel_i][kernel_j] = kernel[kernel_i][kernel_j]
            output[i_count][j_count] = np.max(kernel)
            j_count += 1
        i_count += 1
    return output


def C3(input, kernel, inchannel, outchannel):
    sum_v = 0  # S2中unit的值
    i_count = 0
    j_count = 0  # 被S2卷积出来的值放入C3的位置
    output = np.zeros((10, 10, 16))
    # S2中feature map的下标
    for index in range(0, 15):
        for i in range(0, 14):
            for j in range(0, 14):  # 卷积一次
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j] * kernel[kernel_i][kernel_j]

                output[i_count][j_count][index] = sum_v
                j_count += 1  # C1中的5x5被卷积计算完毕了，可以更新到下一个卷积块了
    i_count += 1  # C1中的一行被卷积计算完毕了，可以更新到下一行了
