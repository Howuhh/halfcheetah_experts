import os
import h5py
import json
import gym
import envs
import pickle
import argparse
import numpy as np

from robomimic.envs.env_gym import EnvGym
from robomimic.utils.log_utils import custom_tqdm


# Script adapted from https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/conversion/convert_d4rl.py
# to work with pickled datasets which was recorded here.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="Env from which data was collected")
    parser.add_argument("--seed", type=int, default=42, help="fix seed for validation demos sampling")
    parser.add_argument("--data_path", type=str, help="path to the data for conversion")
    parser.add_argument("--save_path", type=str, help="path where to save converted data")
    parser.add_argument("--num_valid", type=int, help="number of demos for validation")

    args = parser.parse_args()

    # load input data
    with open(args.data_path, "rb") as data_f:
        data = pickle.load(data_f)
    data_name = args.data_path.split("/")[-1].split('.')[0]

    # create output file
    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, f"{data_name}_converted.hdf5")

    f_sars = h5py.File(output_path, "w")
    f_sars_grp = f_sars.create_group("data")
    f_mask_grp = f_sars.create_group("mask")  # for train/val split

    # code to split D4RL data into trajectories
    # (modified from https://github.com/aviralkumar2907/d4rl_evaluations/blob/bear_intergrate/bear/examples/bear_hdf5_d4rl.py#L18)
    all_obs = data['observations']
    all_act = data['actions']
    N = all_obs.shape[0]

    obs = all_obs[:N-1]
    actions = all_act[:N - 1]
    next_obs = all_obs[1:]
    rewards = np.squeeze(data['rewards'][:N - 1])
    dones = np.squeeze(data['terminals'][:N - 1]).astype(np.int32)

    assert 'timeouts' in data
    timeouts = data['timeouts'][:]

    ctr, total_samples, num_traj = 0, 0, 0
    traj = dict(
        obs=[],
        actions=[],
        rewards=[],
        next_obs=[],
        dones=[]
    )
    env = EnvGym(args.env)

    print("\nConverting hdf5...")
    for idx in custom_tqdm(range(obs.shape[0])):
        # add transition
        traj["obs"].append(obs[idx])
        traj["actions"].append(actions[idx])
        traj["rewards"].append(rewards[idx])
        traj["next_obs"].append(next_obs[idx])
        traj["dones"].append(dones[idx])
        ctr += 1

        # if hit timeout or done is True, end the current trajectory and start a new trajectory
        if timeouts[idx] or dones[idx]:
            # replace next obs with copy of current obs for final timestep, and make sure done is true
            traj["next_obs"][-1] = np.array(obs[idx])
            traj["dones"][-1] = 1

            # store trajectory
            ep_data_grp = f_sars_grp.create_group(f"demo_{num_traj}")
            ep_data_grp.create_dataset("obs/flat", data=np.array(traj["obs"]))
            ep_data_grp.create_dataset("next_obs/flat", data=np.array(traj["next_obs"]))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            ep_data_grp.attrs["num_samples"] = len(traj["actions"])
            total_samples += len(traj["actions"])
            num_traj += 1

            # reset
            ctr = 0
            traj = dict(obs=[], next_obs=[], actions=[], rewards=[], dones=[])

    print(f"\nExcluding {len(traj['actions'])} samples at end of file due to no trajectory truncation.")
    print(f"Wrote {num_traj} trajectories to new converted hdf5 at {output_path}\n")

    # create train/val split masks
    np.random.seed(args.seed)

    valid_ids = np.random.choice(num_traj, size=args.num_valid, replace=False)
    train_ids = [i for i in range(num_traj) if i not in valid_ids]

    print("Validation ids:", valid_ids)
    assert not (set(valid_ids) & set(train_ids))
    f_mask_grp.create_dataset("valid", data=np.array([f"demo_{i}" for i in valid_ids], dtype="S"))
    f_mask_grp.create_dataset("train", data=np.array([f"demo_{i}" for i in train_ids], dtype="S"))

    # metadata
    f_sars_grp.attrs["total"] = total_samples
    f_sars_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
    f_sars.close()
