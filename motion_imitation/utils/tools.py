from khrylib.utils import *


def get_expert(expert_qpos, expert_meta, env):
    old_state = env.sim.get_state()
    expert = {'qpos': expert_qpos, 'meta': expert_meta}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel'}
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:] = qpos
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7])
        ee_pos = env.get_ee_pos(env.cfg.obs_coord)
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat()
        com = env.get_com()
        head_pos = env.get_body_com('head').copy()
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd_new(prev_qpos, qpos, env.dt)
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:3].copy()
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord)
            rangv = qvel[3:6].copy()
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['rq_rmh'].append(rq_rmh)
    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())
    # get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    for key in feat_keys:
        expert[key] = np.vstack(expert[key])
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()
    if expert_meta['cyclic']:
        expert['init_heading'] = get_heading_q(expert_qpos[0, 3:7])
        expert['init_pos'] = expert_qpos[0, :3].copy()
    env.sim.set_state(old_state)
    env.sim.forward()
    return expert