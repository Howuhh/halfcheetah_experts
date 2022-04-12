import os
import gym
import envs
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm.auto import tqdm, trange

from collections import defaultdict
from stable_baselines3 import SAC


def set_seed(env=None, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(data):
    for k, v in data.items():
        if k != "infos":
            data[k] = np.array(v)


def record_episode(env, agent, data=None):
    data = defaultdict(list) if data is None else data

    done, obs, total_reward = False, env.reset(), 0.0
    while not done:
        action, _states = agent.predict(obs, deterministic=True)
        new_obs, reward, done, info = env.step(action)

        timeout = "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
        real_done = done and not timeout

        # record data here (in d4rl format)
        data["observations"].append(obs)
        data["actions"].append(action)
        data["rewards"].append(reward)
        data["timeouts"].append(int(timeout))
        data["terminals"].append(int(real_done))
        data["infos"].append(info)

        # set new state
        obs = new_obs
        total_reward += reward

    tqdm.write(f"Episode done, total reward: {total_reward}")

    return data


def main(config):
    env = gym.make(config.env)
    agent = SAC.load(config.model_path, device=config.device)

    set_seed(env, seed=config.seed)
    data = None
    for _ in trange(config.num_episodes):
        data = record_episode(env, agent, data=data)

    to_numpy(data)

    # just to test shapes
    print("Recorded data shapes:")
    for k, v in data.items():
        if k != "infos":
            print(k, ":", v.shape)
        else:
            print(k, ":", len(v))

    os.makedirs(config.save_path, exist_ok=True)
    with open(os.path.join(config.save_path, env.spec.id + ".pkl"), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or gpu")
    parser.add_argument("--env", type=str, choices=("HalfCheetah-v3", "BackflipCheetah-v0"), help="Environment on which to build a dataset of demo's")
    parser.add_argument("--model_path", type=str, help="Path to the pretrained model checkpoint, should be from stable-baselines3")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect dataset")
    parser.add_argument("--save_path", type=str, help="Full path where to save dataset)")

    main(parser.parse_args())
