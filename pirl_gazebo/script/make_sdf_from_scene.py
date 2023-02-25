import os

import numpy as np
import sys

def _make_box_sdf(save_dir, obs_name, v):
    save_path = save_dir + obs_name + '.sdf'
    f = open(save_path, 'w')
    f.write('<?xml version="1.0" ?>\n')
    f.write('<sdf version="1.5">\n')
    f.write('  <model name="%s">\n' % obs_name)
    f.write('    <pose>%f %f %f %f %f %f</pose>\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))
    f.write('    <static>true</static>\n')
    f.write('    <link name="link">\n')
    f.write('      <inertial>\n')
    f.write('        <mass>1.0</mass>\n')
    f.write('        <inertia> <!-- inertias are tricky to compute -->\n')
    f.write('          <!-- http://gazebosim.org/tutorials?tut=inertia&cat=build_robot -->\n')
    f.write('          <ixx>0.083</ixx>       <!-- for a box: ixx = 0.083 * mass * (y*y + z*z) -->\n')
    f.write('          <ixy>0.0</ixy>         <!-- for a box: ixy = 0 -->\n')
    f.write('          <ixz>0.0</ixz>         <!-- for a box: ixz = 0 -->\n')
    f.write('          <iyy>0.083</iyy>       <!-- for a box: iyy = 0.083 * mass * (x*x + z*z) -->\n')
    f.write('          <iyz>0.0</iyz>         <!-- for a box: iyz = 0 -->\n')
    f.write('          <izz>0.083</izz>       <!-- for a box: izz = 0.083 * mass * (x*x + y*y) -->\n')
    f.write('        </inertia>\n')
    f.write('      </inertial>\n')
    f.write('      <collision name="collision">\n')
    f.write('        <geometry>\n')
    f.write('          <box>\n')
    f.write('            <size>%f %f %f</size>\n' % (v[6], v[7], v[8]))
    f.write('          </box>\n')
    f.write('        </geometry>\n')
    f.write('      </collision>\n')
    f.write('      <visual name="visual">\n')
    f.write('        <geometry>\n')
    f.write('          <box>\n')
    f.write('            <size>%f %f %f</size>\n' % (v[6], v[7], v[8]))
    f.write('          </box>\n')
    f.write('        </geometry>\n')
    f.write('      </visual>\n')
    f.write('    </link>\n')
    f.write('  </model>\n')
    f.write('</sdf>\n')
    f.close()

""" python make_sdf_from_scene.py True """
""" python make_sdf_from_scene.py False {square/s/...} """
if __name__ == '__main__':
    import sys
    FOR_TRAIN = eval(sys.argv[1])
    if FOR_TRAIN:
        for s in range(100):
            scene_txt_file = '/data/pirl_data/eval_show/scene_with_box/scene_{}.txt'.format(s)
            fs = open(scene_txt_file, 'r')
            lines = fs.readlines()
            save_dir = '/data/pirl_data/eval_show/scene_with_box_sdf/{}/'.format(s)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, line in enumerate(lines):
                obs = [float(ele) for ele in line.split(" ")]
                obs_name = 'obs_{}'.format(i)
                _make_box_sdf(save_dir, obs_name, obs)
            fs.close()
    else:
        scene_txt_file = '/data/pirl_data/eval/scene/{}.txt'.format(sys.argv[2])
        fs = open(scene_txt_file, 'r')
        lines = fs.readlines()
        save_dir = '/data/pirl_data/eval/scene_sdf/{}/'.format(sys.argv[2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, line in enumerate(lines):
            obs = [float(ele) for ele in line.split(" ")]
            obs_name = 'obs_{}'.format(i)
            _make_box_sdf(save_dir, obs_name, obs)
        fs.close()
