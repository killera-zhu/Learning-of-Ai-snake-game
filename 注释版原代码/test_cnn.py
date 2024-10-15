import time
#time：用于控制渲染帧之间的延迟。
import random
#random：生成随机种子。见注释5
import torch
#torch：PyTorch库，用于处理深度学习模型。
from sb3_contrib import MaskablePPO
#MaskablePPO：来自Stable Baselines 3的扩展PPO算法，支持动作屏蔽。见注释6
from snake_game_custom_wrapper_cnn import SnakeEnv
#SnakeEnv：自定义的贪吃蛇环境类，用于模拟和渲染游戏。
if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn/ppo_snake_final"
#如果支持多处理器系统（torch.backends.mps.is_available()），则选择使用带有多处理器支持的模型路径。否则，默认选择标准模型路径。

NUM_EPISODE = 10
#NUM_EPISODE：测试的总轮数。
RENDER = True
#RENDER：是否渲染游戏界面。
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
#FRAME_DELAY：渲染每一帧之间的延迟时间。
ROUND_DELAY = 5
#ROUND_DELAY：每一轮游戏结束后的延迟时间

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")
#seed：生成一个随机种子，用于复现实验结果。打印出选定的种子值。

if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

    #根据RENDER变量的值选择是否显示游戏界面。创建一个SnakeEnv的实例作为测试环境。


model = MaskablePPO.load(MODEL_PATH)
#使用Stable Baselines 3中的MaskablePPO类加载预训练的PPO模型。
#MODEL_PATH是之前设定的模型路径，用于加载模型参数和配置。

#开始测试：
total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

#初始化统计变量：
#total_reward、total_score、min_score、max_score：用于存储总体统计信息，如总奖励、总得分、最低得分和最高得分。

#外部循环——测试多轮游戏 对游戏进行初始化
for episode in range(NUM_EPISODE): #循环运行多个测试轮次（NUM_EPISODE次）。
    obs = env.reset() #obs = env.reset()：在每一轮开始时，通过环境的reset()方法重置游戏状态并获取初始观察。
    episode_reward = 0 #episode_reward = 0：初始化本轮的累计奖励。
    done = False #done = False：标志位，表示当前游戏是否结束。
    
    num_step = 0 #num_step = 0：初始化步数计数器，用于记录当前轮游戏进行的步数。
    info = None     #清空蛇蛇内部存储信息

    sum_step_reward = 0 #sum_step_reward = 0：初始化累计步骤奖励，用于记录每轮中每步动作的奖励总和。

    retry_limit = 9 #不太清楚
    print(f"=================== Episode {episode + 1} ==================")
    #显示测试是第几次，用于分隔


#内部循环
    while not done:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        #使用训练好的模型model预测下一个动作action，并根据当前环境状态使用env.get_action_mask()获取动作屏蔽信息。
        prev_mask = env.get_action_mask()
        #存储屏蔽信息 见注释7
        prev_direction = env.game.direction
        #存储蛇的实时移动方向
        num_step += 1
        #步数计数器加一，记录游戏进行的步数。
        obs, reward, done, info = env.step(action)
        #执行预测的动作并观察环境的反馈，返回新的观察状态obs、奖励reward、游戏结束标志done和额外信息info。
        if done:
            if info["snake_size"] == env.game.grid_size:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")
        #如果游戏结束(done == True)：
        #如果蛇的长度等于整个棋盘的大小(info["snake_size"] == env.game.grid_size)，打印胜利信息和相应的奖励。
        #否则，打印游戏结束的惩罚信息和最后一个动作。

        elif info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0
        #如果吃到食物(info["food_obtained"] == True)：
        #打印吃到食物的具体信息，包括奖励和累积步骤奖励，然后重置累积步骤奖励

        else:
            sum_step_reward += reward
        #否则，累加当前步骤的奖励到sum_step_reward中，用于计算累积步骤奖励。


#统计和输出本轮结果
        episode_reward += reward
        #将每个步骤得到的即时奖励 reward 累加到 episode_reward 中，从而得到整个游戏轮次（episode）的总奖励。
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)
        #根据 RENDER 变量的设置来决定是否显示当前游戏画面，以便观察蛇的移动、食物的位置等游戏状态信息。

    episode_score = env.game.score
    #获取当前轮次（episode）结束时游戏环境中的得分，即贪吃蛇游戏中蛇吃到食物的次数。
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    #更新和记录每个轮次游戏的最小和最大得分，以便后续统计和比较。
    
    snake_size = info["snake_size"] + 1
    #获取当前轮次游戏结束时蛇的长度，由于 info["snake_size"] 记录的是蛇身体的长度（不包括头部），所以加上头部即为总长度
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    #打印输出本轮游戏的统计信息，包括轮次编号、累计奖励、得分、总步数和蛇的长度。
    total_reward += episode_reward
    #累加本轮游戏的累计奖励到 total_reward 中，用于计算总体平均奖励。
    total_score += env.game.score
    #累加本轮游戏的得分到 total_score 中，用于计算总体平均得分。
    if RENDER:
        time.sleep(ROUND_DELAY)
    #每个游戏轮次结束后，如果需要渲染游戏画面（即RENDER为True），则程序会暂停ROUND_DELAY秒，以便玩家可以观察到当前轮次游戏的最终状态和统计信息。这种等待时间可以帮助玩家更好地理解和分析游戏过程中的表现和结果。
env.close()
#调用环境的close()方法，关闭游戏环境。
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
#打印测试的总结信息，包括平均得分、最小得分、最大得分和平均奖励，用于评估模型在多轮测试中的表现。