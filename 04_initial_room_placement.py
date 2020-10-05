# Process PoR into Graph with attributes

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

import sys

import inspect
import math

import pygame
from pygame.color import *
from pygame.locals import *

import pymunk
import pymunk as pm
from pymunk.vec2d import Vec2d
import pymunk.pygame_util
import pymunk.autogeometry


def draw_helptext(screen):
    font = pygame.font.Font(None, 20)
    text = ["LMB(hold): Drag shapes",
            "Initial_room_placement_module",
            "P: Pause",
            "During Pause: G: generate iteration file",
            "Esc / Q: Quit",
            "After pressing G, quit module to proceed to Daylight Daylight_hour_optimization_module"
            ]
    y = 5
    for line in text:
        text = font.render(line, 1, THECOLORS["black"])
        screen.blit(text, (5, y))
        y += 15


def draw_graph(graph):
    # draw graph with weight
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos)
    node_labels = nx.get_node_attributes(graph, 'room')
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()
    pass


# ↑ done, draws the graph


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

    # Creating attribute array
    for i in range(len(rooms)):
        Attri_Array[i, i_sizes] = sizes[i]
        Attri_Array[i, i_lights] = lights[i]

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


def site_daylight_map(daylighthours, map_height):
    '''
    site_daylight_map = site_daylight_map(daylighthours, map_height, map_width)
    site_daylight_map = site_daylight_map.tolist()

    These two lines should be used at the end of the code in GH_CPython components,
    to pass arrays among GH_CPython components within grasshopper in the form of list.
    :param daylighthours: "sunlightHoursResult" list received from Ladybug Tools "sunlightHoursAnalysis" component,
    :param map_height:  Number of grid on y axis
    :return: A numpy array of "sunlightHoursResult" that corresponds to the site
    '''
    daylight_map = [daylighthours[i:i + map_height] for i in range(0, len(daylighthours), map_height)]
    np_daylight_map = np.array(daylight_map)
    np_daylight_map = np.rot90(np_daylight_map)  # match the numpy array with the site
    return np_daylight_map
# ↑ done, returns a numpy array, in the form of site grid.




def room_placement_order(graph, rooms):
    # Order of Placement
    lst_sum_weight = []
    lst_degree = []
    for i in range(len(rooms)):
        lst_sum_weight.append(graph.nodes[i]['sum_weight'])
        lst_degree.append(graph.degree[i])

    lst = [lst_sum_weight, lst_degree, sizes, rooms]
    lst_rev = [[] for _ in range(len(lst[0]))]

    for i in range(len(lst)):
        for a in range(len(lst[i])):
            lst_rev[a].append(lst[i][a])

    print(lst_rev)

    d = sorted(lst_rev, key=lambda x: (-x[0], x[1], -x[2]))
    # get the room placement order(in node index)
    order = []
    for i in range(len(d)):
        a = rooms.index(d[i][-1])
        order.append(a)

    return order


# ↑ done, returns a list of node in the order of placement


def add_room_dynamic(space, room_dimension, position):
    '''
    Add a dynamic room to space
    :param space:
    :param room:
    :param room_dimension: Dimension: (width, length); or size: integer.
    :param position: coordinate: (x, y)
    :return: body of the room
    '''
    inf = pm.inf

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

    if type(room_dimension) == 'tuple':
        room_width = room_dimension[0] * 10
        room_length = room_dimension[1] * 10

    else:
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0] * 10
        room_length = ret[1] * 10

    mass = (room_width * room_length) * 0.01
    poly_shape = pm.Poly.create_box(None, size=(room_width, room_length))
    poly_body = pm.Body(mass, inf)
    poly_shape.body = poly_body
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.3
    poly_shape.collision_type = 9
    return poly_body, poly_shape


# ↑ done, add a dynamic room to space


def add_spring(space, a1, a2, weight, length):
    index = 50  # the index to scale the stiffness of the spring according to the weight
    spring = pm.DampedSpring(a1, a2, (0, 0), (0, 0), length, weight * index, 0.3)
    space.add(spring)


def room_placement(screen, space, graph, room_placement_order):
    rooms_shape_list = []
    spring_list = []
    for i in room_placement_order:  # order =  [4, 1, 0, 2, 3]
        s = G.nodes[i]['sizes']
        g = room_placement_order.index(i)
        lst_len = len(room_placement_order)
        exec(f'a{i}, b{i} = add_room_dynamic(space, s, (800+100*(-1)**g, 960 - g*(960/lst_len)), {i})')
        exec(f'space.add(a{i},b{i})')
        exec(f'rooms_shape_list.append(b{i})')
        exec(f'l{i}= (s*100)**0.05')

        f = [n for n in graph.neighbors(i)]  # getting neighbors of node i

        for i_f in range(len(f)):
            for k in room_placement_order[0:g]:
                if k == f[i_f]:
                    weight = G[i][f[i_f]]['weight']  # getting weight of edge connected to node i
                    # if room is inside space:
                    h = f[i_f]
                    sh = G.nodes[h]['sizes']
                    exec(f'l{h} = (sh*100)**0.05')
                    exec(f'length = (l{i} + l{h})')
                    exec(f'add_spring(space, a{h}, a{i}, weight, length)')
                    exec(f'spring_list.append((a{h}, a{i}))')
                else:
                    pass
    return rooms_shape_list, spring_list



def draw_room(screen, rooms_shape_list, spring_list):
    # draw pygame rectangles
    for room in rooms_shape_list:
        i = rooms_shape_list.index(room)
        o = order[i]
        s = G.nodes[o]['sizes']

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

        ret = crack(s)  # crack room size into two closest factors
        room_width = ret[0] * 10
        room_length = ret[1] * 10

        v = room.body.position
        vx = int(v.x - room_width / 2)
        vy = int(v.y - room_length / 2)
        pygame.draw.rect(screen, THECOLORS["red"], (vx, vy, room_width, room_length), 4)

        font = pygame.font.Font(None, 16)
        name = G.nodes[o]['room']
        text = font.render(name, 1, THECOLORS["black"])
        screen.blit(text, (vx, vy))

    for joint in spring_list:
        for i in range(len(spring_list)):
            ax = int(joint[0].position.x)
            ay = int(joint[0].position.y)
            bx = int(joint[1].position.x)
            by = int(joint[1].position.y)
            pygame.draw.line(screen, (0, 0, 255), (ax, ay), (bx, by), 3)


# ↑ done, draw all rooms and connections with pygame


def main():
    # initialize the environment
    pm.pygame_util.positive_y_is_up = False

    pygame.init()
    screen = pygame.display.set_mode((1600, 960))
    clock = pygame.time.Clock()

    space = pm.Space()
    space.gravity = (0.0, 0.0)
    # draw_options = pm.pygame_util.DrawOptions(screen)

    fps = 60

    static = [
        pm.Segment(space.static_body, (0, 0), (0, 960), 5),
        pm.Segment(space.static_body, (0, 960), (1600, 960), 5),
        pm.Segment(space.static_body, (1600, 960), (1600, 0), 5),
        pm.Segment(space.static_body, (0, 0), (1600, 0), 5),
    ]

    for s in static:
        s.collision_type = 0
    space.add(static)

    mouse_joint = None
    mouse_body = pm.Body(body_type=pm.Body.KINEMATIC)

    rooms_shape_list, spring_list = room_placement(screen, space, G, order)

    while True:
        pause = False
        for event in pygame.event.get():
            if event.type == KEYUP:
                if event.key == K_p:
                    pause = True
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                exit()
            elif event.type == MOUSEBUTTONDOWN:
                if mouse_joint != None:
                    space.remove(mouse_joint)
                    mouse_joint = None

                p = Vec2d(event.pos)
                hit = space.point_query_nearest(p, 5, pm.ShapeFilter())
                if hit != None and hit.shape.body.body_type == pm.Body.DYNAMIC:
                    shape = hit.shape
                    # Use the closest point on the surface if the click is outside
                    # of the shape.
                    if hit.distance > 0:
                        nearest = hit.point
                    else:
                        nearest = p
                    mouse_joint = pm.PivotJoint(mouse_body, shape.body,
                                                (0, 0), shape.body.world_to_local(nearest))
                    mouse_joint.max_force = 5000000
                    mouse_joint.error_bias = (1 - 0.15) ** 60
                    space.add(mouse_joint)

            elif event.type == MOUSEBUTTONUP:
                if mouse_joint != None:
                    space.remove(mouse_joint)
                    mouse_joint = None

        while pause == True:
            for event in pygame.event.get():
                if event.type == KEYUP:
                    if event.key == K_p:
                        pause = False
                if event.type == QUIT:
                    exit()
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    exit()
                elif event.type == KEYDOWN and event.key == K_g:  # 判断按下g

                    phase_index = 0
                    rooms_shape_list_pos_out = []
                    body_type_list_out = []
                    room_info_lst = []
                    order_generate = []

                    for shape in rooms_shape_list:
                        i = rooms_shape_list.index(shape)
                        room = order[i]
                        v = shape.body.position
                        vx = int(v.x)
                        vy = int(v.y)
                        body_type = shape.body.body_type
                        shape_pos = (vx, vy)

                        room_info = (room, shape_pos, body_type, i, vy)
                        room_info_lst.append(room_info)

                    order_info_lst = sorted(room_info_lst, key=lambda x: (-x[4], x[3]))

                    for i in range(len(order_info_lst)):
                        room_node = order_info_lst[i][0]
                        shape_pos_order = order_info_lst[i][1]
                        body_type_order = order_info_lst[i][2]
                        order_generate.append(room_node)
                        rooms_shape_list_pos_out.append(shape_pos_order)
                        body_type_list_out.append(body_type_order)

                    f = open('/Old/4_5_info_exchange.txt', mode='w',
                             encoding='utf-8')
                    f.write(f'{phase_index}')
                    f.write('\n')
                    f.write(f'{rooms_shape_list_pos_out}')
                    f.write('\n')
                    f.write(f'{body_type_list_out}')
                    f.write('\n')
                    f.write(f'{order_generate}')
                    f.write('\n')
                    f.write(f'{rooms}')
                    f.write('\n')
                    f.write(f'{sizes}')
                    f.write('\n')
                    f.write(f'{lights}')
                    f.write('\n')
                    f.write(f'{door_direction}')
                    f.write('\n')
                    f.write(f'{circulations}')
                    f.write('\n')
                    f.write(f'{cir_Weight}')
                    print(order_generate)

        screen.fill(pygame.color.THECOLORS["white"])

        draw_room(screen, rooms_shape_list, spring_list)
        draw_helptext(screen)

        mouse_pos = pygame.mouse.get_pos()

        mouse_body.position = mouse_pos

        space.step(1. / fps)

        # space.debug_draw(draw_options)
        pygame.display.flip()

        clock.tick(fps)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))


rooms = ["RoomA", "RoomB", "RoomC", "RoomD", "RoomE"]
sizes = [100, 50, 30, 40, 80]
lights = [8, 7, 6, 5, 4]
door_direction = ["N", "none", "none", "none", "W"]
circulations = [["RoomA", "RoomB", "RoomE", "RoomA"],
                ["RoomB", "RoomD", "RoomC", "RoomE", "RoomB"],
                ["RoomA", "RoomB", "RoomC", "RoomE", "RoomA"]]
cir_Weight = [3, 1, 2]

G = por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight)

order = room_placement_order(G, rooms)

global order_list_out
global rooms_shape_list_out
global spring_list_out

# print("order = ", order)

draw_graph(G)

if __name__ == '__main__':
    sys.exit(main())
