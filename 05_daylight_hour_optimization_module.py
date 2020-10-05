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


def draw_helptext(screen):
    font = pygame.font.Font(None, 20)
    text = ["Daylight_hour_optimization_module",
            "LMB(hold): Drag shapes",
            "P: Pause",
            "During Pause: C: connect spring, then unpause to see reaction",
            "During Pause: G: generate iteration file",
            "Esc / Q: Quit",
            "After final iteration, press G and quit module to proceed"
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


def add_room_static(room_dimension, position):
    '''
    Add a static room to space
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
        room_width = room_dimension[0]*10
        room_length = room_dimension[1]*10

    else:
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0]*10
        room_length = ret[1]*10


    poly_body = pm.Body(body_type=pm.Body.STATIC)
    poly_shape = pm.Poly.create_box(poly_body, size=(room_width, room_length))
    poly_shape.color = (255, 0, 0, 255)  # setting color of the shape
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.7
    poly_shape.collision_type = 2
    return poly_body, poly_shape
# ↑ done, add a static room to space.


def add_room_dynamic(room_dimension, position):
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
        room_width = room_dimension[0]*10
        room_length = room_dimension[1]*10

    else:
        ret = crack(room_dimension)  # crack room size into two closest factors
        room_width = ret[0]*10
        room_length = ret[1]*10

    mass = (room_width * room_length) * 0.01
    poly_shape = pm.Poly.create_box(None, size=(room_width, room_length))
    poly_body = pm.Body(mass, inf)
    poly_shape.body = poly_body
    poly_shape.color = (0, 0, 255, 255)  # setting color of the shape
    poly_body.position = Vec2d(position)
    poly_shape.friction = 0.3
    poly_shape.collision_type = 9
    return poly_body, poly_shape
# ↑ done, add a dynamic room to space


def add_spring(space, a1, a2, weight,length):
    index = 50     # the index to scale the stiffness of the spring according to the weight
    spring = pm.DampedSpring(a1, a2, (0, 0), (0, 0), length, weight * index, 0.3)
    space.add(spring)
# ↑ done, add spring


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


def room_placement_dlh(screen, space, graph, phase_index, rooms_shape_list_pos, body_type_list, room_placement_order):
    # phase_index = infolist[0] + 1   # 0+1 = 1
    # rooms_shape_list_pos = infolist[1]  # [(337, 367), (395, 457), (437, 357), (356, 302), (345, 447)]
    # body_type_list = infolist[2]    #   [0, 0, 0, 0, 0]
    rooms_shape_list = []
    rooms_body_list = []
    connection_list = []
    # body_type_list = [2,0,2,0,2]
    for i in room_placement_order:  # order =  [4, 1, 0, 2, 3]
        g = room_placement_order.index(i)
        pos_x = int(rooms_shape_list_pos[g][0])
        pos_y = int(rooms_shape_list_pos[g][1])
        # pos = (pos_x*1.2, pos_y*1.2)
        s = graph.nodes[i]['sizes']
        name = graph.nodes[i]['room']
        # if body_type_list[g] == 2:
        if g < phase_index-1:
            pos = (pos_x, pos_y)
            exec(f'a{i}, b{i} = add_room_static(s, pos)')
            exec(f'space.add(a{i},b{i})')
            exec(f'rooms_shape_list.append(b{i})')
            exec(f'rooms_body_list.append(a{i})')
        # elif body_type_list[g] == 0:
        else:
            pos = (pos_x, pos_y)
            exec(f'a{i}, b{i} = add_room_dynamic(s, pos)')
            exec(f'space.add(a{i},b{i})')
            exec(f'rooms_shape_list.append(b{i})')
            exec(f'rooms_body_list.append(a{i})')

        exec(f'l{i}= (s*100)**0.05')
        light = graph.nodes[i]['lights']
        if light == 0:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00000000100000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111111)')
        elif light == 1:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00000001000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111110)')
        elif light == 2:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00000010000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111100)')
        elif light == 3:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00000100000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011111000)')
        elif light == 4:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00001000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011110000)')
        elif light == 5:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00010000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011100000)')
        elif light == 6:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b00100000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000011000000)')
        elif light == 7:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b01000000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000010000000)')
        elif light == 8:
            exec(
                f'b{i}.filter = pm.ShapeFilter(categories=0b10000000000000000, mask=pm.ShapeFilter.ALL_MASKS^0b00000000000000000 )')

        f = [n for n in graph.neighbors(i)]  # getting neighbors of node i

        for i_f in range(len(f)):
            for k in room_placement_order[0:g]:
                if k == f[i_f]:
                    # if room is inside space:
                    h = f[i_f]
                    exec(f'connection_list.append((a{h}, a{i}))')
                else:
                    pass
    return rooms_shape_list, rooms_body_list, connection_list


def connect_rooms(space, graph, rooms_body_list, room_placement_order, phase_index):
    spring_list = []
    for body in rooms_body_list:
        g = rooms_body_list.index(body)
        i = room_placement_order[g]
        s = graph.nodes[i]['sizes']
        exec(f'l{g}= (s*100)**0.05')
        f = [n for n in graph.neighbors(i)]  # getting neighbors of node i
        for i_f in range(len(f)):
            for k in room_placement_order[0:g]:
                l = room_placement_order.index(k)
                if k == f[i_f] and (g > phase_index-1 or l > phase_index-1):

                    weight = graph[i][f[i_f]]['weight']  # getting weight of edge connected to node i
                    # if room is inside space:
                    sh = graph.nodes[l]['sizes']
                    exec(f'l{l} = (sh*100)**0.05')
                    exec(f'length = (l{g} + l{l})')
                    exec(f'add_spring(space, rooms_body_list[{l}], rooms_body_list[{g}], weight, length)')
                    exec(f'spring_list.append((rooms_body_list[{l}], rooms_body_list[{g}]))')
                else:
                    pass
    return spring_list



def draw_room(screen, graph, rooms_shape_list, spring_list):
    # draw pygame rectangles
    for room in rooms_shape_list:
        i = rooms_shape_list.index(room)
        o = order[i]
        s = graph.nodes[o]['sizes']
        name = graph.nodes[o]['room']


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
        print(f'{name} pos = {v}')
        vx = int(v.x - room_width / 2)
        vy = int(v.y - room_length / 2)
        pygame.draw.rect(screen, THECOLORS["red"], (vx, vy, room_width, room_length), 2)

        font = pygame.font.Font(None, 16)

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

    fps = 144

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
    # Test code start##########Enable this to test in Pycharm################
    f = open('D:/Delft Courses/Graduation/PythonProject/daylighthour_layout_pathfinding/4_5_info_exchange.txt', mode='r', encoding='utf-8')
    daylighthours = []
    for line in f:
        line = line.strip()
        daylighthours.append(int(line))

    map_height = 96
    map_width = 160
    # Test code end########################################################

    sdmap = site_daylight_map(daylighthours, map_height)
    pixel_list = draw_map(space, sdmap, map_width, map_height)


    # rooms_shape_list, spring_list = room_placement(screen, space, G, order)
    rooms_shape_list, rooms_body_list, connection_list= room_placement_dlh(screen, space, G, phase_index, rooms_shape_list_pos, body_type_list, order)
    spring_list = []







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
                    mouse_joint.max_force = 50000000
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
                elif event.type == KEYDOWN and event.key == K_c:
                    spring_list = connect_rooms(space, G, rooms_body_list, order,phase_index)


                elif event.type == KEYDOWN and event.key == K_g:  # 判断按下g
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

                    f = open('D:/Delft Courses/Graduation/PythonProject/daylighthour_layout_pathfinding/4_5_info_exchange.txt', mode='w',
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

        for shape in rooms_shape_list:
            i = rooms_shape_list.index(shape)
            v = shape.body.position
            x = order[i]
            name = G.nodes[x]['room']
            print(f'{name} pos = ', v)

        screen.fill(pygame.color.THECOLORS["white"])

        draw_map_pg(screen,pixel_list)
        draw_room(screen, G, rooms_shape_list, connection_list)
        draw_helptext(screen)

        mouse_pos = pygame.mouse.get_pos()

        mouse_body.position = mouse_pos

        space.step(1. / fps)

        # space.debug_draw(draw_options)
        pygame.display.flip()

        clock.tick(fps)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))


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

# print("order = ", order)

# draw_graph(G)

if __name__ == '__main__':
    sys.exit(main())