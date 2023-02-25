import numpy as np
import numpy.random as RD

tostring_list = lambda list: ' '.join([str(ele) for ele in list])+'\n'

def write_table(f):
    table = []
    table.append(RD.uniform(0.7, 0.8))
    table.append(RD.uniform(-0.3, 0.3))
    center_z = RD.uniform(0.15, 0.3)
    table.append(center_z)
    table.append(0.0)
    table.append(0.0)
    table.append(0.0)
    table.append(RD.uniform(0.5, 0.6))
    table.append(RD.uniform(0.6, 0.8))
    table.append(center_z*2)
    f.writelines(tostring_list(table))

def write_plane(f):
    plane = []
    cx = RD.uniform(0.8, 1.1)
    cy = RD.uniform(-0.2, 0.2)
    cz = RD.uniform(0.3, 1.1)
    croll = 0.0
    cpitch = 0.0
    cyaw = 0.0
    sx = RD.uniform(0.2, 0.6)
    sy = RD.uniform(0.6, 0.8)
    sz = 0.04
    for i in [cx, cy, cz, croll, cpitch, cyaw, sx, sy, sz]:
        plane.append(i)
    f.writelines(tostring_list(plane))

    if RD.rand() > 0.5:
        back = [cx+sx/2, cy, cz, croll, cpitch, cyaw, 0.04, sy, RD.uniform(sx, 1.0)]
        f.writelines(tostring_list(back))

table_center_xy = [0.8, 0.0]

sample_range_x = [0.27, 0.37]
sample_range_y = [0.5, 1.0]
sample_range_z = [0.4, 0.7]

sub_box_range_x = [0.05, 0.2]
sub_box_range_y = [0.05, 0.2]
sub_box_range_z = [0.05, 0.5]
sub_box_range_yaw = [0.0, 1.57]


for i in range(1000):
    f = open('/data/pirl_data/eval_show/scene_with_box/scene_{}.txt'.format(i), 'w')

    center_x = table_center_xy[0]
    center_y = table_center_xy[1]
    height = RD.uniform(sample_range_z[0], sample_range_z[1])
    center_z = height/2
    center_roll = 0.0
    center_pitch = 0.0
    center_yaw = RD.uniform(-0.2, 0.2)
    box_size_x = RD.uniform(sample_range_x[0], sample_range_x[1]) * 2
    box_size_y = RD.uniform(sample_range_y[0], sample_range_y[1]) * 2
    box_size_z = height
    f.writelines(tostring_list([center_x, center_y, center_z, center_roll, center_pitch, center_yaw, box_size_x, box_size_y, box_size_z]))

    for _ in range(RD.randint(4)):
        bx = RD.uniform(sub_box_range_x[0], sub_box_range_x[1])
        by = RD.uniform(sub_box_range_y[0], sub_box_range_y[1])
        sx = RD.uniform(center_x - box_size_x/2 + bx*np.sqrt(2)/2, center_x + box_size_x/2 - bx*np.sqrt(2)/2)
        sy = RD.uniform(center_y-box_size_y/2 + by*np.sqrt(2)/2, center_y+box_size_y/2 - by*np.sqrt(2)/2)
        sh = RD.uniform(sub_box_range_z[0], sub_box_range_z[1])
        sz = height + sh/2
        cy = RD.uniform(sub_box_range_yaw[0], sub_box_range_yaw[1])
        f.writelines(tostring_list([sx, sy, sz, 0.0, 0.0, cy, bx, by, sh]))
    f.close()
