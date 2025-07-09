import argparse
import os
import pickle

import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-l", "--log_dir", type=str, default="logs")
    parser.add_argument("-p", "--param_name", type=str, default="test")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"{args.log_dir}/{args.exp_name}/{args.param_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    command_cfg["eval"] = True
    command_cfg["lin_vel_x_range"] = [1.0, 1.0]
    command_cfg["lin_vel_y_range"] = [0.0, 0.0]
    command_cfg["ang_vel_range"] = [0.0, 0.0]
    env_cfg["episode_length_s"] = 2
    env_cfg["dof_init_noise"] = 0.0 
    reward_cfg["reward_scales"] = {}
    env_cfg["random_terrain"] = False

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
