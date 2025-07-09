import torch
import math
import genesis as gs
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

np.set_printoptions(threshold=np.inf, linewidth=2000)

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Curriculum:
    def set_to(self, low, high, value=1.0):
        """指定された範囲(low, high)内にあるビンの重みを特定の値(value)に設定する。"""
        # gridの各行がlow以上かつhigh以下であるかを判定
        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0)

        assert len(inds) != 0, "初期化しようとしているコマンド範囲が空です。範囲設定を確認してください。"

        # 範囲内のビンの重みを設定。範囲外は0のままなのでサンプリングされない。
        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)

        self.cfg = cfg = {}
        indices = {}
        for key, v_range in key_ranges.items():
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            # ビンの中心値を計算
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            indices[key] = np.linspace(0, v_range[2]-1, v_range[2])

        self.keys = [*key_ranges.keys()]
        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        # グリッドの作成
        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        
        self._l = l = len(self.grid[0])
        self.indices = np.arange(l)
        
        # 重みを0で初期化
        self.weights = np.zeros(l)

    def sample_bins(self, batch_size, low=None, high=None):
        """重みに基づいてビンの中心(centroid)をサンプリングする"""
        # サンプリング確率を計算
        weight_sum = self.weights.sum()
        if weight_sum == 0:
            # 全ての重みが0なら、一様分布でサンプリング
            probabilities = np.ones_like(self.weights) / len(self.weights)
        else:
            probabilities = self.weights / weight_sum
        inds = self.rng.choice(self.indices, batch_size, p=probabilities)
        # ビンの中心値を返す
        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        """ビンの中心周りの一様分布からサンプリングする"""
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = centroids - bin_sizes / 2, centroids + bin_sizes / 2
        return self.rng.uniform(low, high)

    def sample(self, batch_size, low=None, high=None):
        """最終的なコマンドをサンプリングする"""
        # 1. ビンの中心をサンプリング
        cfg_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        # 2. そのビンの範囲内から一様にサンプリング
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cfg_centroid]), inds
    
class RewardThresholdCurriculum(Curriculum):
    def get_local_bins(self, bin_inds, ranges=0.1):
        """指定されたビンの近傍にあるビンを見つける"""
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        bin_inds = bin_inds.reshape(-1)

        # 中心ビンの座標を取得
        center_coords = self.grid[:, bin_inds, None]
        # 全てのビンとの距離を計算
        # ブロードキャストを利用: (dims, 1, n_bins) と (dims, n_success, 1)
        adjacent_inds = np.logical_and(
            self.grid[:, None, :] >= center_coords - ranges.reshape(-1, 1, 1),
            self.grid[:, None, :] <= center_coords + ranges.reshape(-1, 1, 1)
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.55):
        """パフォーマンスに基づいて重みを更新する"""
        if len(bin_inds) == 0: return
        
        # 全ての報酬が閾値を超えたかをチェック
        is_success_per_task = [(rew / (thresh + 1e-8)) > 1.0 for rew, thresh in zip(task_rewards, success_thresholds)]
        is_success = np.all(is_success_per_task)

        if is_success:
            # 成功した場合、そのビンと近傍のビンの重みを増やす
            target_bins = self.get_local_bins(bin_inds, ranges=local_range).any(axis=0)
            self.weights[target_bins] = np.clip(self.weights[target_bins] + 0.1, 0.0, 1.0)
            print(f"Success! Updated weights for bins: {bin_inds}, grid: {self.grid[:, bin_inds]}")
            print(f"rewards: {task_rewards}, thresholds: {success_thresholds}, is_success: {is_success_per_task}")

    
class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.obs_noise_std = env_cfg.get("obs_noise_std", 0.0)
        self.act_noise_std = env_cfg.get("act_noise_std", 0.0)

        self.simulate_action_latency = False  # there is a 1 step latency on real robot
        self.dt = env_cfg.get("dt", 0.02)  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=env_cfg["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(1.0, 4.0, 0.3),
                camera_lookat=(1.0, 0.0, 0.3),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add ground
        if self.env_cfg.get("random_terrain", False):
            self.ground = self.scene.add_entity(gs.morphs.Terrain(pos=(-16,-16,0.0)))      
        else:
            self.ground = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)) 


        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # カリキュラム学習の初期化
        self._init_curriculum()

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _init_curriculum(self):
        """3つの移動コマンドに対するカリキュラムを初期化する"""
        # <<< 変更: 絶対上限(limit)と初期範囲(initial_range)をConfigから取得
        limits = self.command_cfg["limit"]
        initial_ranges = self.command_cfg["initial_range"]
        bins = self.command_cfg["num_bins"]
        
        # 1. カリキュラムオブジェクトを絶対上限(limit)で初期化
        self.curriculum = RewardThresholdCurriculum(
            seed=123,
            x_vel=(limits["lin_vel_x"][0], limits["lin_vel_x"][1], bins["lin_vel_x"]),
            y_vel=(limits["lin_vel_y"][0], limits["lin_vel_y"][1], bins["lin_vel_y"]),
            yaw_vel=(limits["ang_vel_yaw"][0], limits["ang_vel_yaw"][1], bins["ang_vel_yaw"]),
        )
        
        # 2. set_to を使って、初期のサンプリング範囲を設定
        initial_low = np.array([initial_ranges["lin_vel_x"][0], initial_ranges["lin_vel_y"][0], initial_ranges["ang_vel_yaw"][0]])
        initial_high = np.array([initial_ranges["lin_vel_x"][1], initial_ranges["lin_vel_y"][1], initial_ranges["ang_vel_yaw"][1]])
        self.curriculum.set_to(initial_low, initial_high, value=1.0)
        
        self.env_command_bins = np.zeros(self.num_envs, dtype=np.int32)
        self.command_sums = {
             "tracking_lin_vel": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
             "tracking_ang_vel": torch.zeros(self.num_envs, device=self.device, dtype=torch.float),
        }
        self.curriculum_thresholds = {
            "tracking_lin_vel": self.command_cfg["tracking_lin_vel_sigma"],
            "tracking_ang_vel": self.command_cfg["tracking_ang_vel_sigma"],
        }

    def _resample_commands(self, env_ids):
        """カリキュラムに基づいて、移動コマンドをリサンプリングする"""
        if len(env_ids) == 0: return

        
        if self.command_cfg.get("eval", False):
            # 評価モードでは、コマンドをランダムにリサンプリング
            self.commands[env_ids, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(env_ids),), gs.device)
            self.commands[env_ids, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(env_ids),), gs.device)
            self.commands[env_ids, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(env_ids),), gs.device)
            return
        # ステップ1: パフォーマンス評価
        ep_len = int(self.env_cfg["resampling_time_s"] / self.dt)
        for env_id in env_ids:
            task_rewards = []
            success_thresholds = []
            
            # この環境(env_id)のパフォーマンスを計算
            for key in ["tracking_lin_vel", "tracking_ang_vel"]:
                if key in self.command_sums:
                    reward = self.command_sums[key][env_id] / ep_len
                    reward_scale = reward / self.reward_scales[key]
                    task_rewards.append(reward_scale.item())
                    success_thresholds.append(self.curriculum_thresholds[key])
            
            # この環境が挑戦していた難易度ビンを取得
            old_bin = self.env_command_bins[env_id.cpu().numpy()]
            
            # この環境のパフォーマンスに基づいてカリキュラムを更新
            if len(task_rewards) > 0:
                self.curriculum.update(np.array([old_bin]), task_rewards, success_thresholds)
        
            
        # ステップ3: 新しいコマンドのサンプリング
        new_commands, new_bin_inds = self.curriculum.sample(batch_size=len(env_ids))
        
        self.commands[env_ids, :] = torch.from_numpy(new_commands).float().to(self.device)
        self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds

        # 後処理
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

        

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        if not self.command_cfg.get("eval", False) and self.act_noise_std > 0:
            noise = torch.randn_like(self.actions) * self.act_noise_std
            self.actions = self.actions + noise
            # （必要に応じて再クリップ）
            self.actions = torch.clip(self.actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if name in self.command_sums:
                self.command_sums[name] += rew

        # compute observations
        obs = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        if not self.command_cfg.get("eval", False) and self.obs_noise_std > 0:
 
            # （必要に応じて再クリップ）
            obs[:, :3] = obs[:, :3] + torch.randn_like(obs[:, :3]) * self.obs_noise_std
            obs[:, 9:21] = obs[:, 9:21] + torch.randn_like(obs[:, 9:21]) * self.obs_noise_std
            obs[:, 21:33] = obs[:, 21:33] + torch.randn_like(obs[:, 21:33]) * self.obs_noise_std
        self.obs_buf = obs

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        if self.env_cfg.get("dof_init_noise", 0.0) > 0.0:
            noise_scale = self.env_cfg["dof_init_noise"]
            # envs_idx × num_actions のノイズ行列
            noise = (torch.rand((len(envs_idx), self.num_actions), device=self.device) * 2.0 - 1.0) \
                    * noise_scale
            # 繰り返しテンソルを生成して、default + noise
            base = self.default_dof_pos.unsqueeze(0).repeat(len(envs_idx), 1)
            self.dof_pos[envs_idx] = base + noise
            self.dof_vel[envs_idx] = 0.0
            self.robot.set_dofs_position(
                position=self.dof_pos[envs_idx],
                dofs_idx_local=self.motor_dofs,
                zero_velocity=True,
                envs_idx=envs_idx,
            )
        else:
            self.dof_pos[envs_idx] = self.default_dof_pos
            self.dof_vel[envs_idx] = 0.0
            self.robot.set_dofs_position(
                position=self.dof_pos[envs_idx],
                dofs_idx_local=self.motor_dofs,
                zero_velocity=True,
                envs_idx=envs_idx,
            )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self._resample_commands(envs_idx)
        for key in self.command_sums.keys():
            self.command_sums[key][envs_idx] = 0.0

        # randomization
        self.randomize_friction()
        self.randomize_pd_gains()
        self.randomize_armature()
        self.randomize_com_shift()
        self.randomize_link_properties()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        
        bin_inds = torch.from_numpy(self.env_command_bins).to(self.device)[envs_idx]
        x_idx, y_idx, yaw_idx = torch.unravel_index(bin_inds, (self.command_cfg["num_bins"]["lin_vel_x"], self.command_cfg["num_bins"]["lin_vel_y"], self.command_cfg["num_bins"]["ang_vel_yaw"]))
        self.extras["episode"]["command_lin_vel_x"]   = float(x_idx.float().mean().item())
        self.extras["episode"]["command_lin_vel_y"]   = float(y_idx.float().mean().item())
        self.extras["episode"]["command_ang_vel_yaw"] = float(yaw_idx.float().mean().item())

        nx = self.command_cfg["num_bins"]["lin_vel_x"]
        ny = self.command_cfg["num_bins"]["lin_vel_y"]
        nz = self.command_cfg["num_bins"]["ang_vel_yaw"]
        W = self.curriculum.weights.reshape(nx, ny, nz)

        # X 方向のマージナル
        wx = W.sum(axis=(1,2))
        if wx.sum() > 0:
            px = wx / wx.sum()
        else:
            px = np.ones_like(wx) / len(wx)

        centers_x = self.curriculum.cfg["x_vel"]  # ビン中心の配列
        mean_x = float((centers_x * px).sum())
        var_x  = float(((centers_x - mean_x)**2 * px).sum())

        self.extras["episode"]["weight_mean_lin_vel_x"] = mean_x
        self.extras["episode"]["weight_var_lin_vel_x"]  = var_x

        # 同様に Y, yaw も…
        wy = W.sum(axis=(0,2)); py = wy/wy.sum()
        centers_y = self.curriculum.cfg["y_vel"]
        self.extras["episode"]["weight_mean_lin_vel_y"] = float((centers_y * py).sum())
        self.extras["episode"]["weight_var_lin_vel_y"]  = float(((centers_y - (centers_y*py).sum())**2 * py).sum())

        wz = W.sum(axis=(0,1)); pz = wz/wz.sum()
        centers_z = self.curriculum.cfg["yaw_vel"]
        self.extras["episode"]["weight_mean_ang_vel_yaw"] = float((centers_z * pz).sum())
        self.extras["episode"]["weight_var_ang_vel_yaw"]  = float(((centers_z - (centers_z*pz).sum())**2 * pz).sum())

        # weights = self.curriculum.weights
        # grid = self.curriculum.grid

        # # 非ゼロ重みのビンだけを抽出
        # nonzero_inds = np.nonzero(weights)[0]

        # print(f"{'idx':>4}   {'x':>8}   {'y':>8}   {'yaw':>8}   {'weight':>8}")
        # print("-" * 44)
        # for i in nonzero_inds:
        #     x, y, yaw = grid[:, i]
        #     w = weights[i]
        #     print(f"{i:4d}   {x:8.3f}   {y:8.3f}   {yaw:8.3f}   {w:8.3f}")

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    # ------------ randomization ----------------
    def randomize_link_properties(self):
        # scale mass of links
        mass_scale = 0.9 + 0.2 * torch.rand((self.robot.n_links,), device=self.device)  # 0.9〜1.1
        self.robot.set_links_inertial_mass(mass_scale)

    def randomize_com_shift(self):
        # shift COM positions
        num_links = self.robot.n_links
        link_indices = list(range(num_links))
        com_shift = 0.01 * torch.randn((self.num_envs, num_links, 3), device=self.device)  # +-0.01m=+-10mm
        self.robot.set_COM_shift(com_shift, link_indices)

    def randomize_friction(self):
        # frictions between the ground and robots
        # friction = 0.5 + torch.rand(1).item() # 0.5~1.5
        friction = 0.05 + 1.3*torch.rand(1).item() # 0.2~1.8

        self.robot.set_friction(friction)
        self.ground.set_friction(friction)

    def randomize_pd_gains(self):
        # pd gains of the joint control
        num_dofs = self.robot.n_dofs
        kp_min, kp_max = 18.0, 30.0
        kv_min, kv_max = 0.7, 1.2
        kp = torch.rand(num_dofs, device=self.device) * (kp_max - kp_min) + kp_min
        kv = torch.rand(num_dofs, device=self.device) * (kv_max - kv_min) + kv_min
        self.robot.set_dofs_kp(kp)
        self.robot.set_dofs_kv(kv)

    def randomize_armature(self):
        # joint's rotor inertia
        armature_min, armature_max = 0.01, 0.15
        armature = torch.rand(self.robot.n_dofs, device=self.device) * (armature_max - armature_min) + armature_min
        self.robot.set_dofs_armature(armature)
