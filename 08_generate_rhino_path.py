f = open('/Old/7_path_info_exchange.txt', mode='r', encoding='utf-8')

infolist = []

for line in f:
    infolist.append(line.strip())

path = eval(infolist[0])
x_list = []
y_list = []


for i in range(len(path)):
    x_list_temp = []
    y_list_temp = []
    for j in range(len(path[i])):
        x = path[i][j][0]
        y = 96 - path[i][j][1]
        x_list_temp.append(x)
        y_list_temp.append(y)
    x_list.append(x_list_temp)
    y_list.append(y_list_temp)

len_x = len(x_list)
len_y = len(y_list)

print(x_list)
print(y_list)