"""
https://www.youtube.com/watch?v=JtiK0DOeI4A
"""

import pygame
import math
import numpy as np
import networkx as nx
from queue import PriorityQueue
import pymunk as pm
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

WIDTH = 1600
HEIGHT = 960
WIN = pygame.display.set_mode((WIDTH,HEIGHT))



RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

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

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.width = width
        self.height = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == PURPLE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE


    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid


def draw_grid(win, rows, width, height):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, height))


def draw(win, grid, rows, width, height):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width, height)
    pygame.display.update()


def get_barrier_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


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


def main(win, width,height):
    pygame.init()

    space = pm.Space()
    space.gravity = (0.0, 0.0)
    clock = pygame.time.Clock()
    fps = 120

    ROWS = 160
    grid = make_grid(ROWS, width)



    start = None
    end = None

    run = True
    f = open('/Old/4_5_info_exchange.txt', mode='r', encoding='utf-8')
    infolist = []
    for line in f:
        infolist.append(line.strip())

    rooms_shape_list_ops = eval(infolist[1])
    order = eval(infolist[3])
    rooms = eval(infolist[4])
    sizes = eval(infolist[5])
    lights = eval(infolist[6])
    # door_direction = eval(infolist[7])
    door_direction = ['S', 'S', 'S', 'S', 'S']
    circulations = eval(infolist[8])
    cir_Weight = eval(infolist[9])

    G = por_to_graph(rooms, sizes, lights, door_direction, circulations, cir_Weight)

    print(order)

    start_end_list = []
    stepper = 0
    path_list = []

    for i in order:  # order =  [4, 1, 0, 2, 3]
        s = G.nodes[i]['sizes']
        g = order.index(i)
        lst_len = len(order)
        room_pos = rooms_shape_list_ops[i]
        room_pos_x = rooms_shape_list_ops[i][0]
        room_pos_y = rooms_shape_list_ops[i][1]
        size = G.nodes[i]['sizes']
        d_i = G.nodes[i]['door_direction']

        if type(s) == 'tuple':
            room_width = size[0] * 10
            room_length = size[1] * 10

        else:
            ret = crack(s)
            room_width = ret[0] * 10
            room_length = ret[1] * 10

        room_center = (room_pos_x, room_pos_y)
        room_corner_A = (room_pos_x - room_width / 2, room_pos_y - room_length / 2)
        room_corner_B = (room_pos_x - room_width / 2, room_pos_y + room_length / 2)
        room_corner_C = (room_pos_x + room_width / 2, room_pos_y + room_length / 2)
        room_corner_D = (room_pos_x + room_width / 2, room_pos_y - room_length / 2)
        row_center, col_center = get_barrier_pos(room_center, ROWS, width)
        rowA, colA = get_barrier_pos(room_corner_A, ROWS, width)
        rowB, colB = get_barrier_pos(room_corner_B, ROWS, width)
        rowC, colC = get_barrier_pos(room_corner_C, ROWS, width)
        rowD, colD = get_barrier_pos(room_corner_D, ROWS, width)


        if d_i == 'N':
            row = int(row_center - (rowB - rowA)/2 )
            col = int(col_center)
            spot_i = grid[row][col]

        elif d_i == 'S':
            row = int(row_center + (rowB - rowA) / 2)
            col = int(col_center)
            spot_i = grid[row][col]
        elif d_i == 'W':
            row = int(row_center)
            col = int(col_center - (colD-colA)/2)
            spot_i = grid[row][col]
        elif d_i == 'E':
            row = int(row_center)
            col = int(col_center + (colD - colA) / 2)
            spot_i = grid[row][col]
        else:
            row = int(row_center - (rowB - rowA) / 2)
            col = int(col_center)
            spot_i = grid[row][col]



        f = [n for n in G.neighbors(i)]  # getting neighbors of node i

        for i_f in range(len(f)):
            for k in order[0:g]:
                if k == f[i_f]:
                    weight = G[i][k]['weight']  # getting weight of edge connected to node i
                    sk = G.nodes[k]['sizes']
                    d_k = G.nodes[k]['door_direction']

                    ki = order.index(k)

                    if type(sk) == 'tuple':
                        roomk_width = sk[0] * 10
                        roomk_length = sk[1] * 10

                    else:
                        ret = crack(sk)
                        roomk_width = ret[0] * 10
                        roomk_length = ret[1] * 10

                    room_posk_x = rooms_shape_list_ops[ki][0]
                    room_posk_y = rooms_shape_list_ops[ki][1]

                    roomk_center = (room_posk_x, room_posk_y)
                    roomk_corner_A = (room_posk_x - roomk_width / 2, room_posk_y - roomk_length / 2)
                    roomk_corner_B = (room_posk_x - roomk_width / 2, room_posk_y + roomk_length / 2)
                    roomk_corner_C = (room_posk_x + roomk_width / 2, room_posk_y + roomk_length / 2)
                    roomk_corner_D = (room_posk_x + roomk_width / 2, room_posk_y - roomk_length / 2)
                    rowk_center, colk_center = get_barrier_pos(roomk_center, ROWS, width)
                    rowkA, colkA = get_barrier_pos(roomk_corner_A, ROWS, width)
                    rowkB, colkB = get_barrier_pos(roomk_corner_B, ROWS, width)
                    rowkC, colkC = get_barrier_pos(roomk_corner_C, ROWS, width)
                    rowkD, colkD = get_barrier_pos(roomk_corner_D, ROWS, width)


                    if d_k == 'N':
                        row_d_k = int(rowk_center - (rowkB - rowkA) / 2)
                        col_d_k = int(colk_center)
                        spot_k = grid[row_d_k][col_d_k]

                    elif d_k == 'S':
                        row_d_k = int(rowk_center + (rowkB - rowkA) / 2)
                        col_d_k = int(colk_center)
                        spot_k = grid[row_d_k][col_d_k]
                    elif d_k == 'W':
                        row_d_k = int(rowk_center)
                        col_d_k = int(colk_center - (colkD - colkA) / 2)
                        spot_k = grid[row_d_k][col_d_k]
                    elif d_k == 'E':
                        row_d_k = int(rowk_center)
                        col_d_k = int(colk_center + (colkD - colkA) / 2)
                        spot_k = grid[row_d_k][col_d_k]
                    else:
                        row_d_k = int(rowk_center - (rowkB - rowkA) / 2)
                        col_d_k = int(colk_center)
                        spot_k = grid[row_d_k][col_d_k]

                    start_end_list.append([spot_i, spot_k])
                else:
                    pass

    print(start_end_list)

    while run:
        draw(win, grid, ROWS, width, height)


        for event in pygame.event.get():
            if event.type == KEYUP:
                if event.key == K_p:
                    pause = True
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                exit()

            if pygame.mouse.get_pressed()[0]: # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]: # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    spot_s = start_end_list[stepper][0]
                    start = spot_s
                    spot_s.make_start()
                    spot_e = start_end_list[stepper][1]
                    end = spot_e
                    spot_e.make_end()
                    stepper = stepper + 1
                if event.key == pygame.K_SPACE:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    algorithm(lambda: draw(win, grid, ROWS, width, height), grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

                if event.key == pygame.K_g:
                    temp_list = []
                    for i in range(160):
                        for j in range(96):
                            if grid[i][j].color == PURPLE:
                                temp_list.append((i,j))
                            else:
                                pass
                    path_list.append(temp_list)


                if event.key == pygame.K_o:
                    f = open('/Old/7_path_info_exchange.txt', mode='w', encoding='utf-8')
                    f.write(f'{path_list}')


                if event.key == pygame.K_b:
                    for i in range(len(order)):
                        node = order[i]
                        room_pos = rooms_shape_list_ops[i]
                        room_pos_x = rooms_shape_list_ops[i][0]
                        room_pos_y = rooms_shape_list_ops[i][1]
                        size = G.nodes[node]['sizes']

                        if type(size) == 'tuple':
                            room_width = size[0] * 10
                            room_length = size[1] * 10

                        else:
                            ret = crack(size)
                            room_width = ret[0] * 10
                            room_length = ret[1] * 10

                        room_corner_A = (room_pos_x - room_width / 2, room_pos_y - room_length / 2)
                        room_corner_B = (room_pos_x - room_width / 2, room_pos_y + room_length / 2)
                        room_corner_C = (room_pos_x + room_width / 2, room_pos_y + room_length / 2)
                        room_corner_D = (room_pos_x + room_width / 2, room_pos_y - room_length / 2)
                        rowA, colA = get_barrier_pos(room_corner_A, ROWS, width)
                        rowB, colB = get_barrier_pos(room_corner_B, ROWS, width)
                        rowC, colC = get_barrier_pos(room_corner_C, ROWS, width)
                        rowD, colD = get_barrier_pos(room_corner_D, ROWS, width)
                        # print((rowA, colA),(rowB, colB),(rowC, colC),(rowD, colD))

                        for j in range(int(colB - colA)):
                            for i in range(int(rowD - rowA)):
                                row = int(rowA + i)
                                col = int(colA + j)
                                spot = grid[row][col]
                                spot.make_barrier()


        space.step(1. / fps)




        clock.tick(fps)

        pygame.display.set_caption("fps: " + str(clock.get_fps()))


    pygame.quit()


main(WIN, WIDTH, HEIGHT)

