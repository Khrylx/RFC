import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from khrylib.utils.transformation import quaternion_from_euler
from mujoco_py import load_model_from_path, MjSim
from khrylib.rl.envs.common.mjviewer import MjViewer
from khrylib.mocap.pose import load_amc_file, interpolated_traj
import pickle
import argparse
import glfw

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--amc_id', type=str, default=None)
parser.add_argument('--out_id', type=str, default=None)
parser.add_argument('--model_file', type=str, default="mocap_v2")
parser.add_argument('--mocap_fr', type=int, default=120)
parser.add_argument('--scale', type=float, default=0.45)
parser.add_argument('--dt', type=float, default=1/30.0)
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--fix_feet', action='store_true', default=False)
parser.add_argument('--fix_angle', action='store_true', default=False)
args = parser.parse_args()

model_file = f'khrylib/assets/mujoco_models/{args.model_file}.xml'
model = load_model_from_path(model_file)
sim = MjSim(model)
body_qposaddr = get_body_qposaddr(model)
amc_file = f'data/cmu_mocap/amc/{args.amc_id}.amc'
cyclic = False
cycle_offset = 0.0
offset_z = 0.0


def convert_amc_file():

    def get_qpos(pose):
        qpos = np.zeros_like(sim.data.qpos)
        for bone_name, ind2 in body_qposaddr.items():
            ind1 = bone_addr[bone_name]
            if bone_name == 'root':
                trans = pose[ind1[0]:ind1[0] + 3].copy()
                trans[1], trans[2] = -trans[2], trans[1]
                angles = pose[ind1[0] + 3:ind1[1]].copy()
                quat = quaternion_from_euler(angles[0], angles[1], angles[2])
                quat[2], quat[3] = -quat[3], quat[2]
                qpos[ind2[0]:ind2[0] + 3] = trans
                qpos[ind2[0] + 3:ind2[1]] = quat
            else:
                qpos[ind2[0]:ind2[1]] = pose[ind1[0]:ind1[1]]
        return qpos

    scale = 1 / args.scale * 0.0254
    poses, bone_addr = load_amc_file(amc_file, scale)
    if args.fix_feet:
        poses[:, bone_addr['lfoot'][0] + 2] = poses[:, bone_addr['lfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))
        poses[:, bone_addr['rfoot'][0] + 2] = poses[:, bone_addr['rfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))
    poses_samp = interpolated_traj(poses, args.dt, mocap_fr=args.mocap_fr)
    expert_traj = []
    for cur_pose in poses_samp:
        cur_qpos = get_qpos(cur_pose)
        expert_traj.append(cur_qpos)
    expert_traj = np.vstack(expert_traj)
    expert_traj[:, 2] += args.offset_z
    if args.fix_angle:
        expert_angles = expert_traj[:, 7:]
        while np.any(expert_angles > np.pi):
            expert_angles[expert_angles > np.pi] -= 2 * np.pi
        while np.any(expert_angles < -np.pi):
            expert_angles[expert_angles < -np.pi] += 2 * np.pi
    return expert_traj


def visualize():
    global g_offset, select_start, select_end

    """render or select part of the clip"""
    viewer = MjViewer(sim)
    viewer._hide_overlay = True
    T = 10
    fr = 0
    paused = False
    stop = False

    def find_cyclic_end():
        min_dist = 1e6
        c_end = select_end
        for i in range(select_start + 15, select_end):
            dist = np.linalg.norm(expert_traj[i, 3:] - expert_traj[select_start, 3:])
            if dist < min_dist:
                min_dist = dist
                c_end = i
        offset = expert_traj[c_end + 1, :2] - expert_traj[select_start, :2]
        return c_end, offset

    def key_callback(key, action, mods):
        nonlocal T, fr, paused, stop
        global expert_traj, cyclic, cycle_offset, g_offset, select_start, select_end, offset_z

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            T *= 1.5
        elif key == glfw.KEY_Q:
            exit(0)
        elif key == glfw.KEY_W:
            fr = 0
        elif key == glfw.KEY_E:
            fr = select_end - 1
        elif key == glfw.KEY_Z:
            select_start = fr
            print(f'select start: {select_start}')
        elif key == glfw.KEY_X:
            select_end = fr + 1
            print(f'select end: {select_end}')
        elif key == glfw.KEY_C:
            if select_end > select_start:
                expert_traj = expert_traj[select_start: select_end, :]
                g_offset += select_start
                select_end -= select_start
                select_start = 0
                fr = 0
        elif key == glfw.KEY_V:
            select_end, cycle_offset = find_cyclic_end()
            print(f'cycle_end: {select_end}, cycle_offset: {cycle_offset}')
            cyclic = True
        elif key == glfw.KEY_R:
            stop = True
        elif key == glfw.KEY_D:
            T = max(1, T * 1.5)
        elif key == glfw.KEY_F:
            T = max(1, T / 1.5)
        elif key == glfw.KEY_RIGHT:
            fr = (fr + 1) % expert_traj.shape[0]
            print(fr)
        elif key == glfw.KEY_LEFT:
            fr = (fr - 1) % expert_traj.shape[0]
            print(fr)
        elif key == glfw.KEY_UP:
            offset_z += 0.005
            print(offset_z)
        elif key == glfw.KEY_DOWN:
            offset_z -= 0.005
            print(offset_z)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        else:
            return False

        return True


    viewer.custom_key_callback = key_callback
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -8.0
    viewer.cam.distance = 5.0
    viewer.cam.lookat[2] = 1.0
    t = 0
    while not stop:
        if t >= math.floor(T):
            fr = (fr+1) % expert_traj.shape[0]
            t = 0
        sim.data.qpos[:] = expert_traj[fr]
        sim.data.qpos[2] += offset_z
        sim.forward()
        viewer.cam.lookat[:2] = sim.data.qpos[:2]
        viewer.render()
        if not paused:
            t += 1

    select_start = g_offset + select_start
    select_end = g_offset + select_end
    return select_start, select_end


expert_traj = convert_amc_file()
select_start = 0
select_end = expert_traj.shape[0]
g_offset = 0
if args.render:
    visualize()
print('expert traj shape:', expert_traj.shape)
meta = {'dt': args.dt, 'mocap_fr': args.mocap_fr, 'scale': args.scale, 'offset_z': args.offset_z,
        'cyclic': cyclic, 'cycle_offset': cycle_offset,
        'select_start': select_start, 'select_end': select_end,
        'fix_feet': args.fix_feet, 'fix_angle': args.fix_angle}
print(meta)

"""save the expert trajectory"""
expert_traj_file = f'data/cmu_mocap/motion/{args.out_id}.p'
os.makedirs(os.path.dirname(expert_traj_file), exist_ok=True)
pickle.dump((expert_traj, meta), open(expert_traj_file, 'wb'))


