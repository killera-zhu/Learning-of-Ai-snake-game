import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32#mps（apple GPU加速） 环境数量控制

LOG_DIR = "logs"#日志

os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)#转换字符串为浮点数并且确保大于0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler  #线性插值，就是两个端点中间的数值是多少 就是拿直线近似表达原函数

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])#并行环境

    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="mps",
            verbose=1,
            n_steps=2048,
            batch_size=512*8,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using CUDA.
        model = MaskablePPO(
            "CnnPolicy",#cnn
            env,#环境
            device="cuda",#计算设备
            verbose=1,#输出信息
            n_steps=2048,#每一批次的收集经验步数
            batch_size=512,#每个训练批次的大小
            n_epochs=4,#每个训练批次的更新次数
            gamma=0.94,#折扣因子，*未来奖励
            learning_rate=lr_schedule,#学习率参数
            clip_range=clip_range_schedule,#梯度裁剪参数
            tensorboard_log=LOG_DIR#日志目录
        )

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = "trained_models_cnn"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")#保存频率，保存路径，前缀名

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()