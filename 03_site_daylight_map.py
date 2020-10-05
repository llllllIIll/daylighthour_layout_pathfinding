import numpy as np

daylighthours = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
map_height = 3
map_width = 5

def site_daylight_map(daylighthours, map_height, map_width):
    daylight_map = [daylighthours[i:i + map_height] for i in range(0, len(daylighthours), map_height)]
    # print(daylight_map)
    np_daylight_map = np.array(daylight_map)
    np_daylight_map = np.rot90(np_daylight_map)
    return np_daylight_map

a = site_daylight_map(daylighthours, map_height, None)
print(a)