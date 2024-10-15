import os
import sys
import random

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_mlp import SnakeEnv
#日志和模型保存目录
NUM_ENV = 32
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler （线性插值函数）
def linear_schedule(initial_value, final_value=0.0):
    # 如果initial_value是字符串类型，则将其转换为浮点数类型
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        # 如果final_value是字符串类型，则将其转换为浮点数类型
        final_value = float(final_value)
        # 断言initial_value必须大于0.0
        assert (initial_value > 0.0)

    # 定义一个内部函数scheduler，用于计算进度对应的值
    def scheduler(progress):
        # 根据进度和初始值、最终值计算返回值
        return final_value + progress * (initial_value - final_value)

    # 返回scheduler函数作为结果
    return scheduler
#环境工厂函数
def make_env(seed=0):
    def _init():
        # 创建一个SnakeEnv环境实例，并传入种子值
        env = SnakeEnv(seed=seed)

        # 使用ActionMasker对SnakeEnv环境进行封装，传入SnakeEnv.get_action_mask作为获取动作掩码的方法
        # ActionMasker的作用是对环境进行动作限制，确保只选择合法的动作
        env = ActionMasker(env, SnakeEnv.get_action_mask)

        # 使用Monitor对SnakeEnv环境进行封装，用于记录环境的状态、动作等信息
        env = Monitor(env)

        # 对封装后的环境设置种子值
        env.seed(seed)

        # 返回封装后的环境实例
        return env

    return _init

def main():

    # 为每个环境生成一个随机种子列表
    # Generate a list of random seeds for each environment.
    # 为每个环境生成一个随机种子列表
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.（并行运行）
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])
    # 定义学习率裁剪范围调度器（帮助模型更快地收敛到最优解）
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Instantiate a PPO agent
    # 实例化一个PPO代理的AI模型
    model = MaskablePPO(
        "MlpPolicy",#这是策略网络的类型，MlpPolicy通常表示一个多层感知机（MLP）网络。它用于逼近值函数和/或策略。
        env,#这是你要与之交互的环境的实例。它定义了任务的目标和规则。
        device="cuda",#这指定了计算应该在哪里进行。"cuda"表示使用NVIDIA的CUDA GPU进行计算，这通常可以显著加速计算。
        verbose=1,#这决定了日志信息的详细程度。verbose=1通常意味着会显示训练进度和重要的统计数据。
        n_steps=2048,#这定义了在一个策略更新之前，智能体与环境交互的最大步数。这通常被称为“rollout length”或“trajectory length”。
        batch_size=512,#这定义了用于更新策略的批量大小。批量越大，通常可以更有效地利用计算资源，但也可能需要更多的内存。
        n_epochs=4,#这定义了在每个更新步骤中，通过批量数据训练策略网络的轮数（epochs）。
        gamma=0.94,#这是折扣因子，用于计算累积奖励。它决定了未来奖励在当前决策中的重要性。值越接近1，未来奖励就越重要。
        learning_rate=lr_schedule,#这定义了学习率调度器，它可能是一个函数或对象，用于在训练过程中动态地调整学习率。
        clip_range=clip_range_schedule,#这定义了PPO算法中的clip range，它限制了新旧策略之间的比率，以确保策略更新的稳定性。
        tensorboard_log=LOG_DIR#这指定了TensorBoard日志文件的存储目录。TensorBoard是一个可视化工具，可以帮助你监控和理解训练过程。
    )

    # Set the save directory
    # 设置保存目录（确保在你的当前工作目录下有一个名为trained_models_mlp的目录，如果该目录不存在，则创建它；如果该目录已经存在，则不会报错。）
    save_dir = "trained_models_mlp"
    os.makedirs(save_dir, exist_ok=True)
    #在训练过程中保存模型当前状态的一种方式，以便在需要时可以恢复训练或进行模型评估
    checkpoint_interval = 1000 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    #回调函数，相当于存档点
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    # 将训练日志从标准输出写入文件（sys os 是Python的内置模块）
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        # Start learning
        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        #关闭环境
        env.close()

    
    # Restore stdout（恢复标准输出）
    sys.stdout = original_stdout

    # Save the final model（保存最终模型）
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()
