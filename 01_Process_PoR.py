# Process PoR into Graph with attributes

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

    ReL_Array = ReL_Array + np.rot90(np.fliplr(ReL_Array))
    print("ReL_Array = ", ReL_Array)

    # Creating attribute array
    for i in range(len(rooms)):
        Attri_Array[i, i_sizes] = sizes[i]
        Attri_Array[i, i_lights] = lights[i]
    print("Attri_Array = ", Attri_Array)

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

rooms = ["RoomA", "RoomB", "RoomC", "RoomD", "RoomE"]
sizes = [100, 50, 30, 40, 80]
lights = [3, 2, 1, 2, 4]
door_direction = ["N","none","none","none","W"]
circulations = [["RoomA", "RoomB", "RoomE", "RoomA"],
                ["RoomB", "RoomD", "RoomC", "RoomE", "RoomB"],
                ["RoomA", "RoomB", "RoomC", "RoomE", "RoomA"]]
cir_Weight = [3, 1, 2]

G = por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight)

print(G)

# draw graph with weight
pos = nx.spring_layout(G)
nx.draw(G, pos)
node_labels = nx.get_node_attributes(G, 'room')
nx.draw_networkx_labels(G, pos, labels=node_labels)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
