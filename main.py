import gym
import envs
import imageio
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

DEVICE = "cuda:0"


def rollout(env, model, normalizer=None, render=False):
    obs, done = env.reset(), False
    if normalizer is not None:
        obs = normalizer.normalize_obs(obs)

    total_reward, frames = 0, []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if normalizer is not None:
            obs = normalizer.normalize_obs(obs)

        if render:
            frames.append(env.render(mode="rgb_array"))

    if render:
        imageio.mimsave("rollout.mp4", frames, fps=32)

    return total_reward


def main():
    eval_env = gym.make("BackflipCheetah-v0")
    env = gym.make("BackflipCheetah-v0")

    # env = gym.make("HalfCheetah-v3")
    # eval_env = gym.make("HalfCheetah-v3")

    # model = SAC(
    #     "MlpPolicy", env,
    #     learning_rate=3e-4,
    #     tau=5e-3,
    #     batch_size=256,
    #     learning_starts=1000,
    #     buffer_size=1000000,
    #     tensorboard_log="logs",
    #     verbose=1,
    #     seed=42,
    #     device=DEVICE
    # )
    # model.learn(
    #     total_timesteps=3_000_000,
    #     callback=CheckpointCallback(save_freq=100_000, save_path="logs/sac_forward_checkpoints"),
    #     eval_env=eval_env,
    #     eval_freq=100_000,
    #     n_eval_episodes=10
    # )
    # model.save("sac_forward_backflip")

    model = SAC.load("sac_backflip", device=DEVICE)
    # model = SAC.load("sac_forward", device=DEVICE)
    print(rollout(eval_env, model, render=True))


if __name__ == "__main__":
    main()