def convert(lst):
    return eval(lst)



f = open('D:/Delft Courses/Graduation/PythonProject/daylighthour_layout_pathfinding/4_5_info_exchange.txt', mode='r', encoding='utf-8')
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
door_direction = eval(infolist[7])
circulations = eval(infolist[8])
cir_Weight = eval(infolist[9])





print(phase_index)
print(rooms_shape_list_ops)
print(body_type_list)
print(order)
print(rooms)
print(sizes[1])
print(lights)
print(door_direction[1])
print(circulations[1])
print(cir_Weight[1])
#
#
# f = open('D:\Delft Courses\Graduation\PythonProject\Old\daylighthours.txt', mode='r', encoding='utf-8')
# daylighthour = []
# for line in f:
#     line  = line.strip()
#     daylighthour.append(int(line))
