import yaml
import os
import glob
import numpy as np
from khrylib.utils import recreate_dirs


class Config:

    def __init__(self, cfg_id, test, create_dirs=False, cfg_dict=None):
        self.id = cfg_id
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = 'motion_imitation/cfg/**/%s.yml' % cfg_id
            files = glob.glob(cfg_path, recursive=True)
            assert(len(files) == 1)
            cfg = yaml.safe_load(open(files[0], 'r'))
        # create dirs
        base_dir = '/tmp' if test else 'results'
        self.base_dir = os.path.expanduser(base_dir)

        self.cfg_dir = '%s/motion_im/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        self.video_dir = '%s/videos' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)

        # expert
        self.motion_id = cfg['motion_id']
        self.expert_traj_file = f'data/cmu_mocap/motion/{self.motion_id}.p'

        # training config
        self.gamma = cfg.get('gamma', 0.95)
        self.tau = cfg.get('tau', 0.95)
        self.policy_htype = cfg.get('policy_htype', 'relu')
        self.policy_hsize = cfg.get('policy_hsize', [300, 200])
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weightdecay = cfg.get('policy_weightdecay', 0.0)
        self.value_htype = cfg.get('value_htype', 'relu')
        self.value_hsize = cfg.get('value_hsize', [300, 200])
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weightdecay = cfg.get('value_weightdecay', 0.0)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.log_std = cfg.get('log_std', -2.3)
        self.fix_std = cfg.get('fix_std', False)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('min_batch_size', 50000)
        self.mini_batch_size = cfg.get('mini_batch_size', self.min_batch_size)
        self.max_iter_num = cfg.get('max_iter_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.save_model_interval = cfg.get('save_model_interval', 100)
        self.reward_id = cfg.get('reward_id', 'quat')
        self.reward_weights = cfg.get('reward_weights', None)
        self.end_reward = cfg.get('end_reward', False)

        # adaptive parameters
        self.adp_iter_cp = np.array(cfg.get('adp_iter_cp', [0]))
        self.adp_noise_rate_cp = np.array(cfg.get('adp_noise_rate_cp', [1.0]))
        self.adp_noise_rate_cp = np.pad(self.adp_noise_rate_cp, (0, self.adp_iter_cp.size - self.adp_noise_rate_cp.size), 'edge')
        self.adp_log_std_cp = np.array(cfg.get('adp_log_std_cp', [self.log_std]))
        self.adp_log_std_cp = np.pad(self.adp_log_std_cp, (0, self.adp_iter_cp.size - self.adp_log_std_cp.size), 'edge')
        self.adp_policy_lr_cp = np.array(cfg.get('adp_policy_lr_cp', [self.policy_lr]))
        self.adp_policy_lr_cp = np.pad(self.adp_policy_lr_cp, (0, self.adp_iter_cp.size - self.adp_policy_lr_cp.size), 'edge')
        self.adp_noise_rate = None
        self.adp_log_std = None
        self.adp_policy_lr = None

        # env config
        self.mujoco_model_file = '%s.xml' % cfg['mujoco_model']
        self.vis_model_file = '%s.xml' % cfg['vis_model']
        self.env_start_first = cfg.get('env_start_first', False)
        self.env_init_noise = cfg.get('env_init_noise', 0.0)
        self.env_episode_len = cfg.get('env_episode_len', 200)
        self.env_term_body = cfg.get('env_term_body', 'head')
        self.env_expert_trail_steps = cfg.get('env_expert_trail_steps', 0)
        self.obs_type = cfg.get('obs_type', 'full')
        self.obs_coord = cfg.get('obs_coord', 'root')
        self.obs_phase = cfg.get('obs_phase', True)
        self.obs_heading = cfg.get('obs_heading', False)
        self.obs_vel = cfg.get('obs_vel', 'full')
        self.root_deheading = cfg.get('root_deheading', False)
        self.action_type = cfg.get('action_type', 'position')

        # virutual force
        self.residual_force = cfg.get('residual_force', False)
        self.residual_force_scale = cfg.get('residual_force_scale', 200.0)
        self.residual_force_mode = cfg.get('residual_force_mode', 'implicit')
        self.residual_force_bodies = cfg.get('residual_force_bodies', 'all')
        self.residual_force_torque = cfg.get('residual_force_torque', True)

        # joint param
        if 'joint_params' in cfg:
            jparam = zip(*cfg['joint_params'])
            jparam = [np.array(p) for p in jparam]
            self.jkp, self.jkd, self.a_ref, self.a_scale, self.torque_lim = jparam[1:6]
            self.a_ref = np.deg2rad(self.a_ref)
            jkp_multiplier = cfg.get('jkp_multiplier', 1.0)
            jkd_multiplier = cfg.get('jkd_multiplier', jkp_multiplier)
            self.jkp *= jkp_multiplier
            self.jkd *= jkd_multiplier
            torque_limit_multiplier = cfg.get('torque_limit_multiplier', 1.0)
            self.torque_lim *= torque_limit_multiplier

        # body param
        if 'body_params' in cfg:
            bparam = zip(*cfg['body_params'])
            bparam = [np.array(p) for p in bparam]
            self.b_diffw = bparam[1]

    def update_adaptive_params(self, i_iter):
        cp = self.adp_iter_cp
        ind = np.where(i_iter >= cp)[0][-1]
        nind = ind + int(ind < len(cp) - 1)
        t = (i_iter - self.adp_iter_cp[ind]) / (cp[nind] - cp[ind]) if nind > ind else 0.0
        self.adp_noise_rate = self.adp_noise_rate_cp[ind] * (1-t) + self.adp_noise_rate_cp[nind] * t
        self.adp_log_std = self.adp_log_std_cp[ind] * (1-t) + self.adp_log_std_cp[nind] * t
        self.adp_policy_lr = self.adp_policy_lr_cp[ind] * (1-t) + self.adp_policy_lr_cp[nind] * t
