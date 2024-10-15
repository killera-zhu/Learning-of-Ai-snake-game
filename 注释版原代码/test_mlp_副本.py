import time
import random
from sb3_contrib import MaskablePPO
from snake_game_custom_wrapper_mlp import SnakeEnv
#环境配置
# 设置模型保存路径
MODEL_PATH = r"trained_models_mlp/ppo_snake_final"
# 设置测试时运行的回合数
NUM_EPISODE = 10
# 设置是否渲染游戏界面，如果为True，则可以看到游戏运行的过程
RENDER = True
# 设置每一帧的延迟时间，控制游戏运行的速度
# 0.01 表示较快的速度，0.05 表示较慢的速度
FRAME_DELAY = 0.05 
# 设置每轮游戏开始前的延迟时间，可能是为了让玩家观察或等待
ROUND_DELAY = 5
# 生成一个随机种子，用于确保测试的可重复性
# 这里种子值范围设置为0到1e9之间
seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

#环境初始化
# 如果RENDER为True，则创建SnakeEnv环境，并设置不限制步数（limit_step=False），并且不处于静默模式（silent_mode=False），
# 这样可以看到游戏界面和游戏过程。
if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)

# 如果RENDER为False，则创建SnakeEnv环境，同样不限制步数，但设置为静默模式（silent_mode=True），
# 这样游戏界面将不会显示，程序将在没有可视化的情况下运行。
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

# 加载已训练好的模型
# 这里从指定的路径加载使用MaskablePPO算法训练得到的模型
# MaskablePPO是一个支持maskable actions的PPO变种
model = MaskablePPO.load(MODEL_PATH)

# 初始化总奖励和总分数变量，用于记录整个测试过程中的奖励和得分
total_reward = 0
total_score = 0

# 初始化最小得分和最大得分变量，用于记录测试过程中的最好和最坏成绩
min_score = 1e9  # 初始化为一个较大的数，确保后续能被正确更新为最小的分数
max_score = 0     # 初始化为0，确保后续能被正确更新为最大的分数
      
# 遍历每一个回合（episode）
for episode in range(NUM_EPISODE):
    # 重置环境，获取初始观察值
    obs = env.reset()
    # 初始化当前回合的奖励为0
    episode_reward = 0
    # 标记游戏是否结束
    done = False
    
    # 记录当前回合的步数
    num_step = 0
    # 用于存储环境返回的额外信息
    info = None

    # 记录连续步骤的奖励总和（在某些游戏中，可能每步的奖励较小，需要累加多个步骤的奖励来观察效果）
    sum_step_reward = 0

    # 设定重试次数限制（在实际情况中可能不直接使用，但在某些场景下可以用来限制游戏内重试次数）
    retry_limit = 9
    # 打印当前回合的标题
    print(f"=================== Episode {episode + 1} ==================")

    # 步数计数器（在一些复杂场景下，可能与num_step有所不同）
    step_counter = 0
    # 游戏循环，直到游戏结束
    while not done:
        # 使用模型预测动作，并考虑环境的动作掩码（mask）
        action, _ = model.predict(obs, action_masks=env.get_action_mask())

        # 记录前一个状态的动作掩码和游戏方向
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction

        # 更新步数
        num_step += 1

        # 执行动作，获取新的观察值、奖励、游戏是否结束以及额外信息
        obs, reward, done, info = env.step(action)

        # 如果游戏结束
        if done:
            # 将动作索引转换为对应的方向字符串
            last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
            # 打印游戏结束时的惩罚和最后一步的动作
            print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        # 如果获得了食物
        elif info["food_obtained"]:
            # 打印获得食物时的步数、食物奖励和连续步骤的奖励总和
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            # 重置连续步骤的奖励总和
            sum_step_reward = 0

        # 其他情况（即没有获得食物）
        else:
            # 累加连续步骤的奖励
            sum_step_reward += reward

        # 累加当前回合的奖励
        episode_reward += reward

        # 如果设置了RENDER，则渲染游戏界面并暂停一段时间
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    # 获取当前回合的得分
    episode_score = env.game.score
    # 更新最小得分
    if episode_score < min_score:
        min_score = episode_score
    # 更新最大得分
    if episode_score > max_score:
        max_score = episode_score

    # 打印当前回合的奖励总和、得分、总步数和蛇的长度
    snake_size = info["snake_size"] + 1  # 假设info中的snake_size是蛇的节数，这里加1得到蛇的总长度
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    # 累加总奖励和总得分
    total_reward += episode_reward
    total_score += env.game.score

    # 如果设置了RENDER，则在游戏回合之间暂停一段时间
    if RENDER:
        time.sleep(ROUND_DELAY)
# 关闭环境，释放资源
env.close()

# 打印测试总结的标题
print(f"=================== Summary ==================")

# 打印平均得分，即将所有回合的得分总和除以回合数
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")

# 解释：
# 1. env.close()：关闭游戏环境，释放掉可能占用的系统资源，如内存、图形渲染资源等。
# 2. 打印了一个标题“Summary”，用来分隔前面的游戏运行记录与后面的统计信息。
# 3. 打印了平均得分、最小得分、最大得分和平均奖励。平均得分是所有回合得分的平均值，最小得分是所有回合中的最低分，最大得分是所有回合中的最高分，平均奖励是所有回合中获得的奖励的平均值。