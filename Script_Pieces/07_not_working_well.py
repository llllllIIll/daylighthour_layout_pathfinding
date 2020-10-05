# 需要进行抽象化的有：节点（属性有：x y坐标  父节点  g及h ）  地图（属性有高度 宽度  数据(数据中有可通行路径与障碍两种)）
# A_star :
# open_list (存放待测试的点，刚开始有start加入list，后面每一个current的相邻点（不能位于close_list中且不是终点）要放到open_list中) 表示已经走过的点
# close_list(存放已测试的点，已经当过current的点，就放到close_list中) 存放已经探测过的点，不必再进行探测
# current 现在正在测试的点，要计算current周围的点的代价f  经过current后要放到close_list中 将openlist代价f最小的node当作下一个current
# start_point end_point

# 初始化地图  openlist closelist node
# 将start点放入openlist中
# while（未达到终点）：
# 取出 openlist 中的点 将这个点设置为current 并放入closelist中
# for node_near in（current的临近点）
# if（current的临近点 node_near 不在closelist中且不为障碍）：
# 计算 node_near 的f（f=g+h）大小
# if( node_near 不在 openlist 中)
# 将 node_near 放入 openlist，并将其父节点设置为current 然后将f值设置为计算出的f值
# else if( node_near 在 openlist 中)
#  if（计算出的f大于在openlist中的f）
#    不动作
#  else if（计算出的f小于等于在openlist中的f）
#     将 openlist 中的 node_near 的f值更新为计算出的新的更小的f值 并将父节点设置为current
# 返回并继续循环
import sys


# 将地图中的点抽象化成类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):  # 函数重载
        if ((self.x == other.x) and (self.y == other.y)):
            return 1
        else:
            return 0


# 通过列表实现的地图的建立
class map_2d:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.data = []
        self.data = [[0 for i in range(width)] for j in range(height)]

    def map_show(self):
        for j in range(self.height):
            for i in range(self.width):
                print(self.data[j][i], end=' ')
            print("")

    def obstacle(self, obstacle_x, obstacle_y):
        self.data[obstacle_x][obstacle_y] = 1

    def end_draw(self, point):
        self.data[point.x][point.y] = 6


# A*算法的实现
class A_star:
    # 设置node
    class Node:
        def __init__(self, point, endpoint, g):
            self.point = point  # 自己的坐标
            self.endpoint = endpoint  # 自己的坐标
            self.father = None  # 父节点
            self.g = g  # g值，g值在用到的时候会重新算
            self.h = (abs(endpoint.x - point.x) + abs(endpoint.y - point.y)) * 10  # 计算h值
            self.f = self.g + self.h

        # 寻找临近点
        def search_near(self, ud, rl):  # up  down  right left
            nearpoint = Point(self.point.x + rl, self.point.y + ud)
            nearnode = A_star.Node(nearpoint, self.endpoint, self.g + 1)
            return nearnode

    def __init__(self, start_point, end_point, map):  # 需要传输到类中的，在此括号中写出
        self.path = []
        self.close_list = []  # 存放已经走过的点
        self.open_list = []  # 存放需要尽心探索的点
        self.current = 0  # 现在的node
        self.start_point = start_point
        self.end_point = end_point
        self.map = map  # 所在地图

    def select_current(self):
        min = 10000000
        node_temp = 0
        for ele in self.open_list:
            if ele.f < min:
                min = ele.f
                node_temp = ele
        self.path.append(node_temp)
        self.open_list.remove(node_temp)
        self.close_list.append(node_temp)
        return node_temp

    def isin_openlist(self, node):
        for opennode_temp in self.open_list:
            if opennode_temp.point == node.point:
                return opennode_temp
        return 0

    def isin_closelist(self, node):
        for closenode_temp in self.close_list:
            if closenode_temp.point == node.point:
                return 1
        return 0

    def is_obstacle(self, node):
        if self.map.data[node.point.x][node.point.y] == 1:
            return 1
        return 0

    def near_explore(self, node):
        ud = 0
        rl = 1
        node_temp = node.search_near(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.open_list.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.open_list.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.open_list.append(node_temp)

        ud = 0
        rl = -1
        node_temp = node.search_near(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.open_list.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.open_list.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.open_list.append(node_temp)
        ud = 1
        rl = 0
        node_temp = node.search_near(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.open_list.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.open_list.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.open_list.append(node_temp)

        ud = -1
        rl = 0
        node_temp = node.search_near(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.open_list.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.open_list.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.open_list.append(node_temp)

        return 0


def draw_obstacle(map, G, order, rooms_shape_list_ops):
    def room_obstacle(center_pos_x, center_pos_y, width, length):
        mp_width = int(length/10)
        mp_length = int(width/10)
        for len in range(mp_length):
            for wid in range(mp_width):
                obstacle_x = int(wid + center_pos_x/10  - width/10 / 2)
                obstacle_y = int(len + center_pos_y/10  - length/10 / 2)
                map.obstacle(96 - obstacle_y, obstacle_x)

    for room in order:
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

        i = order.index(room)
        node = order[i]
        s = G.nodes[node]['sizes']
        ret = crack(s)  # crack room size into two closest factors
        room_width = ret[1] * 10
        room_length = ret[0] * 10
        pos = rooms_shape_list_ops[i]
        pos_x = pos[0]
        pos_y = 960 - pos[1]
        room_obstacle(pos_x, pos_y, room_width, room_length)


def door_pair(G, order, rooms_shape_list_ops):
    def door_position(room_node, center_pos_x, center_pos_y, width, length):
        direction = G.nodes[room_node]['door_direction']
        if direction == 'N':
            door_x = int(center_pos_x/10 )
            door_y = int(center_pos_y/10  + length/10 / 2)
        elif direction == 'S':
            door_x = int(center_pos_x/10)
            door_y = int(center_pos_y/10 - length/10 / 2)
        elif direction == 'E':
            door_x = int(center_pos_x/10 + width/10 / 2)
            door_y = int(center_pos_y/10)
        elif direction == 'W':
            door_x = int(center_pos_x/10 - width/10 / 2)
            door_y = int(center_pos_y/10)
        else:
            door_x = int(center_pos_x/10)
            door_y = int(center_pos_y/10 + length/10/ 2)
        # door_x = int(center_pos_x / 10)
        # door_y = int(center_pos_y / 10)
        door_pos = (door_x, door_y)
        return door_pos

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

    pair_list = []

    f = [e for e in G.edges]
    for i in range(len(f)):

        a = f[i][0]
        g = order.index(a)
        sa = G.nodes[a]['sizes']
        ret = crack(sa)  # crack room size into two closest factors
        a_room_width = ret[1] * 10
        a_room_length = ret[0] * 10
        pos_a = rooms_shape_list_ops[g]
        pos_a_x = pos_a[0]
        pos_a_y = 960 - pos_a[1]
        start_pos = door_position(a, pos_a_x, pos_a_y, a_room_width, a_room_length)

        b = f[i][1]
        k = order.index(b)
        sb = G.nodes[b]['sizes']
        ret = crack(sb)  # crack room size into two closest factors
        b_room_width = ret[1] * 10
        b_room_length = ret[0] * 10
        pos_b = rooms_shape_list_ops[k]
        pos_b_x = pos_b[0]
        pos_b_y = 960 - pos_b[1]
        end_pos = door_position(b, pos_b_x, pos_b_y, b_room_width, b_room_length)
        pair_list.append([start_pos, end_pos])


    # for room in order:
    #
        # pair_temp = []
        # g = order.index(room)
        # node = order[g]
        # s = G.nodes[node]['sizes']
        # ret = crack(s)  # crack room size into two closest factors
        # room_width = ret[1] * 10
        # room_length = ret[0] * 10
        # pos_g = rooms_shape_list_ops[g]
        # pos_g_x = pos_g[0]
        # pos_g_y = 960 - pos_g[1]
        # start_pos = door_position(node, pos_g_x, pos_g_y, room_width, room_length)
        # f = [n for n in G.neighbors(node)]  # getting neighbors of node i
    #
    #     for i_f in range(len(f)):
    #         for k in order[0:g]:
    #             if k == f[i_f]:
    #                 # weight = G[node][f[k]]['weight']  # getting weight of edge connected to node i
    #                 sk = G.nodes[k]['sizes']
    #                 ret = crack(sk)  # crack room size into two closest factors
    #                 room_width = ret[0] * 10
    #                 room_length = ret[1] * 10
    #
    #                 pos_k = rooms_shape_list_ops[k]
    #                 pos_k_x = pos_k[0]
    #                 pos_k_y = pos_k[1]
    #                 print((pos_g_x,pos_g_y),(pos_k_x, pos_k_y))
    #
    #                 end_pos = door_position(k, pos_k_x, pos_k_y, room_width, room_length)
    #                 pair_temp.append(start_pos)
    #                 pair_temp.append(end_pos)
    #                 pair_list.append(pair_temp)
    #                 pair_temp = []
    #                 print(start_pos,end_pos)
    #             else:
    #                 pass

    return pair_list



import numpy as np
import networkx as nx


def convert(lst):
    return eval(lst)


def por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight):
    i_sizes = 0
    i_lights = 1
    i_weight_sum = 2
    i_edge_sum = 3
    i_door_direction = 4

    ReL_Array = np.zeros([len(rooms), len(rooms)])
    Attri_Array = np.zeros([len(rooms), i_door_direction + 1])

    # creating ReL_Array
    for i in range(len(rooms)):
        for a in range(len(circulations)):
            for b in range(len(circulations[a]) - 1):
                if circulations[a][b] == rooms[i]:
                    ReL_Array[i, rooms.index(circulations[a][b + 1])] += cir_Weight[a]

    # Flip the array left and right then rotate the array by 90 degrees
    ReL_Array = ReL_Array + np.rot90(np.fliplr(ReL_Array))
    # print("ReL_Array = ", ReL_Array)

    # Creating attribute array
    for i in range(len(rooms)):
        Attri_Array[i, i_sizes] = sizes[i]
        Attri_Array[i, i_lights] = lights[i]
    # print("Attri_Array = ", Attri_Array)

    # get Sum of edge weights on one node
    sum_weight = []
    for i in range(len(rooms)):
        sum_weight.append(sum(ReL_Array[i]))

    # Creating graph in networkx from ReL_Array
    G = nx.from_numpy_array(ReL_Array)
    for i in range(len(rooms)):
        G.nodes[i]['room'] = rooms[i]
        G.nodes[i]['sizes'] = sizes[i]
        G.nodes[i]['lights'] = lights[i]
        G.nodes[i]['sum_weight'] = sum_weight[i]
        G.nodes[i]['door_direction'] = door_direction[i]

    print("G.edges = ", G.edges(data=True))  # print edges
    print("G.nodes = ", G.nodes(data=True))  # print all nodes with attribute dictionary
    print("G.degree() = ", G.degree())
    print("sum_weight = ", sum_weight)
    return G


f = open('/Old/4_5_info_exchange.txt', mode='r', encoding='utf-8')
infolist = []
for line in f:
    infolist.append(line.strip())

phase_index = infolist[0]
rooms_shape_list_ops = eval(infolist[1])
body_type_list = eval(infolist[2])
order = eval(infolist[3])
rooms = eval(infolist[4])
sizes = eval(infolist[5])
lights = eval(infolist[6])
# door_direction = eval(infolist[7])
door_direction = ['S','S','S','S','S']
circulations = eval(infolist[8])
cir_Weight = eval(infolist[9])

G = por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight)

##建图并设立障碍
mw = 96
mh = 160
ss = map_2d(mw, mh)

cx_list = []
cy_list = []
m_list = []
point_con = []

draw_obstacle(ss,G, order, rooms_shape_list_ops)
ss.map_show()

# set path finding pairing points
pair_list = door_pair(G, order, rooms_shape_list_ops)
print(pair_list)

h = len(pair_list)
print(h)

for i in range(h):
    point_s = Point(pair_list[i][0][0], 96 - pair_list[i][0][1])
    point_e = Point(pair_list[i][1][0], 96 - pair_list[i][1][1])
    point_con.append([point_s, point_e])


for i in range(len(point_con)):
    start_point = point_con[i][0]
    end_point = point_con[i][1]
    ss.end_draw(end_point)
    ss.end_draw(start_point)

    a_star = A_star(start_point, end_point, ss)  # 初始化设置A*
    start_node = a_star.Node(start_point, end_point, 0)
    a_star.open_list.append(start_node)

    flag = 0  # 到达终点的标志位
    m = 0  # 步数统计
    cx = 0
    cy = 0

    while flag != 1:  # 进入循环
        a_star.current = a_star.select_current()  # 从openlist中选取一个node
        flag = a_star.near_explore(a_star.current)  # 对选中的node进行周边探索
        m = m + 1

    # print("Step =", m)
    m_list.append(m + 1)

    for node_path in a_star.path:  # 画出地图路径
        ss.end_draw(node_path.point)
        cx = node_path.point.x
        cy = node_path.point.y
        cx_list.append(cx)
        cy_list.append(cy)
        # print(cx, ",", cy)
    cx_list.append(end_point.x)
    cy_list.append(end_point.y)

jj = map_2d(mw, mh)

# set obstacle
draw_obstacle(jj, G, order, rooms_shape_list_ops)

for i in range(len(cx_list)):
    jj.obstacle(cy_list[i], cx_list[i])

jj.map_show()

print(order)
print(door_direction)