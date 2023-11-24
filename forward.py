import numpy as np
from PIL import Image
from init import InitKernel_C1

def c1(input, kernel, outchannel):
    sum_v = 0  # S2中unit的值
    i_count = 0
    j_count = 0  # 被C1卷积出来的值放入S2的位置
    output = np.zeros((28, 28, 6))
    for index in range(0, outchannel):  # S2中feature map的下标
        for i in range(0, 32):
            for j in range(0, 32):  # 卷积一次
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j] * kernel[kernel_i][kernel_j][index]
                output[i_count][j_count][index] = sum_v
                j_count += 1  # C1中的5x5被卷积计算完毕了，可以更新到下一个卷积块了
        i_count += 1  # C1中的一行被卷积计算完毕了，可以更新到下一行了
    return output


def s2(input, kernel):
    i_count = 0
    j_count = 0
    output = np.zeros((14, 14, 6))
    for index in range(0, 6):
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                for kernel_i in range(i, i + 2):
                    for kernel_j in range(j, j + 2):
                        input[kernel_i][kernel_j] = kernel[kernel_i][kernel_j]
                output[i_count][j_count] = np.max(kernel)
                j_count += 1
            i_count += 1
    return output


def c3(input, kernel):
    sum_v = 0  # S2中unit的值
    i_count = 0
    j_count = 0  # 被S2卷积出来的值放入C3的位置
    output = np.zeros((10, 10, 16))
    # S2中feature map的下标
    # 前六组，对三个feature map进行卷积
    a = 0  # 前六个feature map,每个feature map都是取上一层feature map的其中三个进行卷积并求和再加上一个权重
    k_index = 0  # kernel_index, total 60
    for index in range(0, 6):
        b = (a + 1) % 6
        c = (b + 1) % 6
        for i in range(0, 14):
            for j in range(0, 14):  # 卷积一次
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j][a] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][b] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][c] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                output[i_count][j_count][index] = sum_v
                sum_v = 0  # sum_v要存储下一个数了
                j_count += 1  # S2中的5x5被卷积计算完毕了，可以更新到下一个卷积块了
            i_count += 1  # S2中的一行被卷积计算完毕了，可以更新到下一行了
        a = (a + 1) % 6

    # 后六组，四个feature map进行卷积
    sum_v = 0  # 更新sum_v的值
    a = 0  # 前六个feature map,每个feature map都是取上一层feature map的其中三个进行卷积并求和再加上一个权重
    for index in range(6, 12):
        b = (a + 1) % 6
        c = (b + 1) % 6
        d = (c + 1) % 6
        for i in range(0, 14):
            for j in range(0, 14):  # 卷积一次
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j][a] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][b] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][c] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][d] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                output[i_count][j_count][index] = sum_v
                sum_v = 0
                j_count += 1  # S2中的5x5被卷积计算完毕了，可以更新到下一个卷积块了
            i_count += 1  # S2中的一行被卷积计算完毕了，可以更新到下一行了
        a = (a + 1) % 6

    # 接下来三组， 选择四个不相邻的feature map

    sum_v = 0
    a = 0

    for index in range(12, 15):  # 计算的是feature map,例如这里会进行三次循环,那么结果就会产生3个feature map
        b = (a + 1) % 6
        c = (b + 2) % 6
        d = (c + 1) % 6
        for i in range(0, 14):
            for j in range(0, 14):
                for kernel_i in range(i, i + 5):
                    for kernel_j in range(j, j + 5):
                        sum_v += input[kernel_i][kernel_j][a] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1  # a层input和k_index层kernel卷积
                        sum_v += input[kernel_i][kernel_j][b] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][c] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                        sum_v += input[kernel_i][kernel_j][d] * kernel[kernel_i][kernel_j][k_index]
                        k_index += 1
                output[i_count][j_count][index] = sum_v
                sum_v = 0
                j_count += 1
            i_count += 1
        a = (a + 1) % 6

    # 最后一层feature map, 会用上前面所有6层feature map
    for i in range(0, 14):
        for j in range(0, 14):
            for kernel_i in range(i, i + 5):
                for kernel_j in range(j, j + 5):
                    sum_v += input[kernel_i][kernel_j][0] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1  # a层input和k_index层kernel卷积
                    sum_v += input[kernel_i][kernel_j][1] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1
                    sum_v += input[kernel_i][kernel_j][2] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1
                    sum_v += input[kernel_i][kernel_j][3] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1
                    sum_v += input[kernel_i][kernel_j][4] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1
                    sum_v += input[kernel_i][kernel_j][5] * kernel[kernel_i][kernel_j][k_index]
                    k_index += 1
            output[i_count][j_count][15] = sum_v
    return output


def s4(input, kernel):
    i_count = 0
    j_count = 0
    output = np.zeros((5, 5, 16))
    for index in range(0, 16):
        for i in range(0, 10, 2):
            for j in range(0, 10, 2):
                for kernel_i in range(i, i + 2):
                    for kernel_j in range(j, j + 2):
                        input[kernel_i][kernel_j][index] = kernel[kernel_i][kernel_j][index]
                output[i_count][j_count] = np.max(kernel)
                j_count += 1
            i_count += 1
    return output


def c5(input, kernel, inchannel, outchannel):
    k_index = 0
    output = np.zeros((1, 1, 120))
    for num in range(0, outchannel):
        sum_v = 0
        for k in range(0, 16):
            for i in range(0, 5):
                for j in range(0, 5):
                    sum_v += input[i][j][k] * kernel[i][j][k_index]
                    k_index += 1
        output[0][0][num] = sum_v
        return output


def f6():
    pass


def output():
    pass

weight = InitKernel_C1()
if __name__ == '__main__':
    img = Image.open('../0.jpg').convert('L')
    img = np.array(img).reshape((32, 32))
    o1 = c1(img, weight, 6)
    print(o1)
