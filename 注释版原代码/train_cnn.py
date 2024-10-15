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

#导入必要的Python模块和库，包括操作系统功能（os）、系统相关功能（sys）、随机数生成（random）、PyTorch深度学习库（torch）、
#稳定基线（Stable Baselines）库中的监控功能（Monitor）、向量化环境（SubprocVecEnv）、回调函数（CheckpointCallback）、
#MaskablePPO算法（MaskablePPO）以及自定义的贪吃蛇环境（SnakeEnv）和相关的包装器（ActionMasker）。具体介绍见注释8

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
# 根据是否支持Metal Performance Shaders（MPS），设定并行环境的数量 NUM_ENV。 见注释9
LOG_DIR = "logs"
#设置日志目录 LOG_DIR，LOG_DIR是一个字符串变量，用于指定存储训练日志和TensorBoard数据的目录路径。
os.makedirs(LOG_DIR, exist_ok=True)
#创建文件

#定义了一个线性调度器函数 linear_schedule，用于在训练过程中线性调整学习率和剪切范围。
def linear_schedule(initial_value, final_value=0.0):
    #initial_value: 初始值，可以是浮点数或字符串（如果是字符串，则会尝试转换为浮点数）。
    #final_value: 最终值，默认为0.0。调度器将在进度从0到1之间线性地将值从initial_value降低到final_value
    #见注释10

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0) #确保initial_value的值大于0。
#类型检查与转换：如果initial_value是字符串，则将其转换为浮点数。同时将final_value也转换为浮点数，以确保后续计算的一致性。

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)
    return scheduler
#生成调度器函数：scheduler(progress)
    #函数接受一个progress参数，用于表示当前的进度或时间步。通常，progress的取值范围在[0, 1]之间，表示训练或优化过程的进展程度。
    #在函数内部，根据线性插值的方式，计算并返回一个在给定进度下的调度值。//不太理解



def make_env(seed=0): #seed是一个可选的参数，默认值为0，用于设置SnakeEnv环境的随机种子。在强化学习中，随机种子的设置可以影响训练的随机性和可重复性。
    def _init():
        env = SnakeEnv(seed=seed)
        #使用给定的随机种子创建SnakeEnv环境。SnakeEnv是一个自定义的贪吃蛇游戏环境，用于训练强化学习代理程序
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        #使用ActionMasker包装器，将SnakeEnv环境中的动作空间进行屏蔽（masking）。这样可以限制代理程序在每个状态下可以选择的动作，以便在不同情况下实现特定的行为限制或策略。
        env = Monitor(env)
        #将Monitor监控器应用于SnakeEnv环境。Monitor用于记录和可视化环境的训练统计信息，如奖励值、步数等，以便后续分析和评估训练效果。
        env.seed(seed)
        #将指定的随机种子应用于SnakeEnv环境，确保在不同运行中具有相同的初始状态，从而增强训练过程的可重复性和控制性。
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    #创建一个空集合，用于存储随机种子。
    while len(seed_set) < NUM_ENV:  #循环直到集合中的随机种子数量达到 NUM_ENV 的设定值。
        seed_set.add(random.randint(0, 1e9))    #随机生成一个范围在 0 到 10^9 之间的整数，并将其添加到 seed_set 集合中。确保每个环境实例都有不同的随机种子。
    #生成足够多数量的种子

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    #创建一个并行化的环境实例，其中每个环境都是通过调用make_env(seed=s)函数生成的。

#MPS可用时的配置
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
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    #这段代码的目的是根据系统硬件环境选择合适的硬件加速选项，以优化 PPO 模型在 Snake 环境中的训练效率和性能。

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = "trained_models_cnn"
    os.makedirs(save_dir, exist_ok=True)
    #设置保存模型的目录 根据 MPS 的可用性选择保存模型的目录，如果支持 MPS，则保存在 "trained_models_cnn_mps"，否则保存在 "trained_models_cnn"。

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    #指定了保存检查点的间隔步数，每隔15625步保存一次检查点 见注释11
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")
    #CheckpointCallback 是一个回调函数，用于在训练过程中定期保存模型的检查点
    #save_freq=checkpoint_interval 指定了保存检查点的频率。
    #save_path=save_dir 指定了保存检查点的路径。
    #name_prefix="ppo_snake" 给保存的模型文件名添加前缀，以便识别和区分不同的模型。



    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt") #将训练过程中的标准输出重定向到文件 "training_log.txt" 中，以便记录和保存训练日志。
    with open(log_file_path, 'w') as log_file: #文件读写
        sys.stdout = log_file #sys.stdout = log_file 将标准输出流重定向到指定的日志文件中。

        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        #开始执行模型的训练过程，total_timesteps=int(100000000) 指定了总的训练步数。
        #通过回调函数 checkpoint_callback 定期保存模型的检查点。
        env.close()

    # Restore stdout
    sys.stdout = original_stdout # 恢复标准输出

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))
    #使用 model.save() 方法将最终训练好的模型保存为压缩文件 "ppo_snake_final.zip"，保存路径为之前设置的保存目录中。

if __name__ == "__main__":
    main()
#确保主函数在直接运行该脚本时被调用，开始执行整个训练流程。