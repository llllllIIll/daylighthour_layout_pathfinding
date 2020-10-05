import numpy as np
import networkx as nx

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
# ↑ done, returns a graph contains all information, weights on edges, attributes on nodes

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






f = open('/Old/4_5_info_exchange.txt', mode='r', encoding='utf-8')
infolist = []
for line in f:
    infolist.append(line.strip())

phase_index = int(infolist[0]) + 1   # 0+1 = 1
rooms_shape_list_pos = eval(infolist[1])  # [(337, 367), (395, 457), (437, 357), (356, 302), (345, 447)]
body_type_list = eval(infolist[2])    #   [0, 0, 0, 0, 0]
order = eval(infolist[3])
rooms = eval(infolist[4])
sizes = eval(infolist[5])
lights = eval(infolist[6])
door_direction = eval(infolist[7])
circulations = eval(infolist[8])
cir_Weight = eval(infolist[9])

G = por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight)


ax_list = []
ay_list = []
bx_list = []
by_list = []
cx_list = []
cy_list = []
dx_list = []
dy_list = []
name_list = []

for node in order[0:phase_index-1]:
    i = order.index(node)
    name = G.nodes[node]['room']
    name_list.append(name)
    size = G.nodes[node]['sizes']
    if type(size) == 'tuple':
        room_width = size[0]
        room_length = size[1]

    else:
        ret = crack(size)  # crack room size into two closest factors
        room_width = ret[0]
        room_length = ret[1]
    pos_x = rooms_shape_list_pos[i][0]/10
    pos_y = (960-rooms_shape_list_pos[i][1])/ 10

    cornerA = (pos_x - room_width/2, pos_y - room_length/2, 0)
    cornerB = (pos_x - room_width / 2, pos_y + room_length / 2, 0)
    cornerC = (pos_x + room_width / 2, pos_y + room_length / 2, 0)
    cornerD = (pos_x + room_width / 2, pos_y - room_length / 2, 0)
    ax_list.append(cornerA[0])
    ay_list.append(cornerA[1])
    bx_list.append(cornerB[0])
    by_list.append(cornerB[1])
    cx_list.append(cornerC[0])
    cy_list.append(cornerC[1])
    dx_list.append(cornerD[0])
    dy_list.append(cornerD[1])






