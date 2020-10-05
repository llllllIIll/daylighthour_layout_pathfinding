# Process PoR into Graph with attributes

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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


def from_list_to_array(list, array_width):
    '''
    Arrays are passed in the form of one dimensional list, each time receiving the one dimensional list,
    it needs to be transformed back to two dimensional array.
    :param list: Array received
    :param array_width: Original width of numpy array
    :return: The original numpy array
    '''
    array = [list[i:i + array_width] for i in range(0, len(list), array_width)]
    array = np.array(array)
    return array
# ↑ done, used for passing arrays among GH_CPython components within grasshopper.


def room_placement_order(graph, rooms):
    # Order of Placement
    lst_sum_weight = []
    lst_degree = []
    lst_size = []
    for i in range(len(rooms)):
        lst_sum_weight.append(graph.nodes[i]['sum_weight'])
        lst_degree.append(graph.degree[i])

    lst = [lst_sum_weight, lst_degree, sizes, rooms]

    lst_rev = [[] for _ in range(len(lst[0]))]
    # print('lst_rev = ', lst_rev)
    for i in range(len(lst)):
        for a in range(len(lst[i])):
            lst_rev[a].append(lst[i][a])

    d = sorted(lst_rev, key=lambda x: (-x[0], x[1], -x[2]))
    # print("d = ", d)

    # get the room placement order(in node index)
    order = []
    for i in range(len(d)):
        a = rooms.index(d[i][-1])
        order.append(a)

    return order
# ↑ done, returns a list of node in the order of placement


def add_room_static(space, room_dimension, position, i):
    '''
    Add a static room to space
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

    if type(room_dimension) == "int":
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0]
        room_length = ret[1]
    else:
        room_width = room_dimension[0]
        room_length = room_dimension[1]

    # poly_body = pm.Body(body_type=pm.Body.STATIC)
    exec(f'poly_body{i} = pm.Body(body_type=pm.Body.STATIC)')
    # poly_shape = pm.Poly.create_box(poly_body, size=(room_width, room_length))
    exec(f'poly_shape{i} = pm.Poly.create_box(poly_body{i}, size=(room_width, room_length))')
    # poly_shape.color = (255, 0, 0, 255)  # setting color of the shape
    exec (f'poly_shape{i}.color = (255, 0, 0, 255)')
    # poly_body.position = Vec2d(position)
    exec(f'poly_body{i}.position = Vec2d(position)')
    # poly_shape.friction = 0.7
    exec(f'poly_shape{i}.friction = 0.7')
    # space.add(poly_body, poly_shape)
    exec (f'space.add(poly_body{i}, poly_shape{i})')
    # poly_shape.collision_type = 2
    exec(f'poly_shape{i}.collision_type = 2')
    # return poly_body
    return exec(f'poly_body{i}')
# ↑ done, add a static room to space. TODO, add name to shape, add outline to shape.


def add_room_dynamic(space, room_dimension, position, i):
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

    ret = crack(room_dimension)  # crack room size into two closest factors
    room_width = ret[0]*10
    room_length = ret[1]*10

    # if is_integer(room_dimension):
    #     ret = crack(room_dimension)  # crack room size into two closest factors
    #     room_width = ret[0]*10
    #     room_length = ret[1]*10
    # else:
    #     room_width = room_dimension[0]
    #     room_length = room_dimension[1]

    mass = (room_width * room_length) * 0.01
    poly_shape = pm.Poly.create_box(None, size=(room_width, room_length))
    poly_body = pm.Body(mass, inf)
    poly_shape.body = poly_body
    # poly_shape.color = (123, 123, 123, 0)
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.3
    poly_shape.collision_type = 9
    return poly_body, poly_shape
# ↑ done, add a dynamic room to space


def add_room_static2(space, room_dimension, position):
    '''
    Add a static room to space
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

    if type(room_dimension) == "int":
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0]
        room_length = ret[1]
    else:
        room_width = room_dimension[0]
        room_length = room_dimension[1]

    poly_body = pm.Body(body_type=pm.Body.STATIC)
    poly_shape = pm.Poly.create_box(poly_body, size=(room_width, room_length))
    poly_shape.color = (255, 0, 0, 255)  # setting color of the shape
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.7
    space.add(poly_body, poly_shape)
    poly_shape.collision_type = 2
    return poly_body
# ↑ done, add a static room to space. TODO, add name to shape, add outline to shape.


def add_room_dynamic2(space, room_dimension, position):
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

    if type(room_dimension) == "int":
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0]
        room_length = ret[1]
    else:
        room_width = room_dimension[0]
        room_length = room_dimension[1]
    mass = (room_width * room_length) * 0.05
    poly_shape = pm.Poly.create_box(None, size=(room_width, room_length))
    poly_body = pm.Body(mass, inf)
    poly_shape.body = poly_body
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.7
    space.add(poly_body, poly_shape)
    poly_shape.collision_type = 9
    return
# ↑ done, add a dynamic room to space


def add_spring(space, a1, a2, weight,length):
    index = 50     # the index to scale the stiffness of the spring according to the weight
    spring = pm.DampedSpring(a1, a2, (0, 0), (0, 0), length, weight * index, 0.3)
    space.add(spring)


def draw_map(space, site_daylight_map, map_width, map_height):
    def draw_pixel(space, position, lighthour):
        poly_body = pm.Body(body_type=pm.Body.STATIC)
        poly_shape = pm.Poly.create_box(poly_body, size=(10, 10))
        poly_body.position = Vec2d(position)
        poly_shape.friction = 0.7
        if lighthour < 1:
            poly_shape.color = (0, 0, 255, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000000001, mask=pm.ShapeFilter.ALL_MASKS^0b00000000100000000)
        elif lighthour < 2:
            poly_shape.color = (67, 0, 189, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000000010, mask=pm.ShapeFilter.ALL_MASKS^0b00000001100000000)
        elif lighthour < 3:
            poly_shape.color = (134, 0, 122, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000000100, mask=pm.ShapeFilter.ALL_MASKS^0b00000011100000000)
        elif lighthour < 4:
            poly_shape.color = (201, 0, 55, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000001000, mask=pm.ShapeFilter.ALL_MASKS^0b00000111100000000)
        elif lighthour < 5:
            poly_shape.color = (255, 12, 0, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000010000, mask=pm.ShapeFilter.ALL_MASKS^0b00001111100000000)
        elif lighthour < 6:
            poly_shape.color = (255, 79, 0, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000000100000, mask=pm.ShapeFilter.ALL_MASKS^0b00011111100000000)
        elif lighthour < 7:
            poly_shape.color = (255, 146, 0, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000001000000, mask=pm.ShapeFilter.ALL_MASKS^0b00111111100000000)
        elif lighthour < 8:
            poly_shape.color = (255, 213, 0, 255)
            poly_shape.filter = pm.ShapeFilter(categories=0b00000000010000000, mask=pm.ShapeFilter.ALL_MASKS^0b01111111100000000)
        space.add(poly_body, poly_shape)
        return poly_body

    pixel_list = []

    for i in range(map_height - 1):
        for j in range(map_width - 1):
            if site_daylight_map[i][j] < 8:
                draw_pixel(space, (j * 10, i * 10), site_daylight_map[i][j])
                a = (j*10, i*10, site_daylight_map[i][j])
                pixel_list.append(a)

    return pixel_list
# ↑ done, draw map into the space


def draw_map_pg(screen,pixel_list):
    for pixel in pixel_list:
        vx = int(pixel[0] - 5)
        vy = int(pixel[1] - 5)
        lighthour = int(pixel[2])
        if lighthour < 1:  # setting color of the shape
            pygame.draw.rect(screen, (0, 0, 255), (vx, vy, 10, 10))
        elif lighthour < 2:
            pygame.draw.rect(screen, (67, 0, 189), (vx, vy, 10, 10))
        elif lighthour < 3:
            pygame.draw.rect(screen, (134, 0, 122), (vx, vy, 10, 10))
        elif lighthour < 4:
            pygame.draw.rect(screen, (201, 0, 55), (vx, vy, 10, 10))
        elif lighthour < 5:
            pygame.draw.rect(screen, (255, 12, 0), (vx, vy, 10, 10))
        elif lighthour < 6:
            pygame.draw.rect(screen, (255, 79, 0), (vx, vy, 10, 10))
        elif lighthour < 7:
            pygame.draw.rect(screen, (255, 146, 0), (vx, vy, 10, 10))
        elif lighthour < 8:
            pygame.draw.rect(screen, (255, 213, 0), (vx, vy, 10, 10))
# # ↑ done, draw map into the space




def corridor(graph, room_placement):
    pass
# ↑ WIP, calculate the shortest route??, locate doors, line into segments, pump rooms away.


def room_placement(screen, space, graph, room_placement_order):
    '''
    初始形体生成模块：
    - 空间里生成房间，连上spring
    - 输出所有房间位置，形体信息，spring设置信息，发给日照优化模块
    :param space:
    :param graph:
    :param room_placement_order:
    :return:
    '''
    rooms_shape_list = []
    spring_list = []
    for i in room_placement_order:  # order =  [4, 1, 0, 2, 3]
        s = G.nodes[i]['sizes']
        exec(f'a{i}, b{i} = add_room_dynamic(space, s, (200+ i*100,200 +i*100), {i})')
        exec(f'space.add(a{i},b{i})')
        # exec(f'print(a{i}.position)')
        # exec(f'print(b{i}.get_vertices())')
        exec(f'rooms_shape_list.append(b{i})')
        exec(f'l{i}= (s*100)**0.05')
        light = G.nodes[i]['lights']
        if light == 0:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00000000100000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111111)')
        elif light == 1:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00000001000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111110)')
        elif light ==2:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00000010000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111100)')
        elif light ==3:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00000100000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111000)')
        elif light ==4:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00001000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011110000)')
        elif light ==5:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00010000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011100000)')
        elif light ==6:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b00100000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011000000)')
        elif light ==7:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b01000000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000010000000)')
        elif light ==8:
            exec(f'b{i}.filter = pm.ShapeFilter(categories=0b10000000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000000000000 )')

        f = [n for n in graph.neighbors(i)] # getting neighbors of node i
        g = room_placement_order.index(i)
        for i_f in range(len(f)):
            for k in room_placement_order[0:g]:
                if k == f[i_f]:
                    weight = G[i][f[i_f]]['weight']    # getting weight of edge connected to node i
                     # if room is inside space:
                    h = f[i_f]
                    sh = G.nodes[h]['sizes']
                    exec(f'l{h} = (sh*100)**0.05')
                    exec(f'length = (l{i} + l{h})')
                    exec (f'add_spring(space, a{h}, a{i}, weight, length)')
                    exec(f'spring_list.append((a{h}, a{i}))')
                else:
                    pass
    return  rooms_shape_list, spring_list
# ↑ WIP,    输出所有房间名称，位置，形体信息，给rhino模块；房间名称，位置，形体信息，spring设置信息，发给日照优化模块
#   room_name_list = []
#   room_body_list = []
#   room_size_list = []
#   room_type_list = []
#   room_position_list = []
#   spring_room_list = []
#   spring_stiffness_list = []


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




def daylight_hour_optimizer():
    '''
    - 输入日照分析地图，输入所有房间的信息，输入被分析的房间的信息
    - 设定被分析房间为STATIC，根据日照分析地图生成STATIC，挤走其余的DYNAMIC
    - 输出所有房间位置，形体信息，spring设置信息
    - 之后所有的信息传输在日照优化模块和Ladybug模块之间进行，只不过迭代时接受的房间位置，形体信息，spring设置信息来自自己。
    - 两个按键，一个用来输出形体信息，一个用来结束优化，输出最终房间位置，形体信息，到走廊生成模块。
    :return:
    '''
# ↑ WIP,


def corridor_generator():
    '''
    - 输入房间位置，形体信息，流线信息。
    - 使用path finding算法生成走廊
    - 使用GA优化开门位置和走廊走向（该部分在P5优化）
    :return:
    '''


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

    # Draw daylight map in to space
    sdmap = site_daylight_map(daylighthours, map_height)
    pixel_list = draw_map(space, sdmap, map_width, map_height)

    rooms_shape_list, spring_list = room_placement(screen, space, G, order)




    while True:
        for event in pygame.event.get():
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
                    mouse_joint.max_force = 50000
                    mouse_joint.error_bias = (1 - 0.15) ** 60
                    space.add(mouse_joint)

            elif event.type == MOUSEBUTTONUP:
                if mouse_joint != None:
                    space.remove(mouse_joint)
                    mouse_joint = None

        screen.fill(pygame.color.THECOLORS["white"])

        draw_map_pg(screen,pixel_list)
        draw_room(screen, rooms_shape_list, spring_list)

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

# print("order = ", order)

# draw_graph(G)


if __name__ == '__main__':
    sys.exit(main())

