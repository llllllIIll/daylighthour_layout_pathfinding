# https://blog.csdn.net/qq_36607894/article/details/103595912
# 把一个数分解为最接近的两个数的乘积
import numpy as np
def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False

ret = crack(40)  # 250, 256
print(ret)
a = ret[0]
b = ret[1]
print(a, b)