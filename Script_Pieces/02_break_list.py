li = [i for i in range(10)]
n = 4   #大列表中几个数据组成一个小列表
li2 = [li[i:i + n] for i in range(0, len(li), n)]
print(li2)