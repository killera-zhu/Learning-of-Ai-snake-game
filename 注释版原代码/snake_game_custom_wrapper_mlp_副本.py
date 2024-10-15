
#引入一些必要的库
import math
import time # For debugging.

import gym
import numpy as np

from snake_game import SnakeGame

class SnakeEnv(gym.Env):
    #初始化蛇游戏以及步数限制
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        #调用父类（或称为基类、超类）的构造函数，以确保父类中的任何初始化操作都被执行
        super().__init__()
        # 初始化蛇游戏
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        # 初始化动作空间，包含四个动作：上、左、右、下
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    
        # 初始化观测空间，表示游戏界面的状态
        # 0: 空，0.5: 蛇身，1: 蛇头，-1: 食物
        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.game.board_size, self.game.board_size),
            dtype=np.float32
        ) # 0: empty, 0.5: snake body, 1: snake head, -1: food

        # 初始化游戏界面大小
        self.board_size = board_size
        # 蛇的最大长度，即游戏界面的格子数的平方
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        # 初始化蛇的长度
        self.init_snake_size = len(self.game.snake)
        # 蛇的最大增长长度，即游戏界面的格子数减去初始蛇的长度
        self.max_growth = self.grid_size - self.init_snake_size

        # 游戏是否结束的标志
        self.done = False

        if limit_step:
            # 如果限制步数，则设置步数限制为游戏界面格子数的四倍
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            # 如果不限制步数，则步数限制设为一个非常大的数
            self.step_limit = 1e9 # Basically no limit.
        # 奖励步数计数器
        self.reward_step_counter = 0

    #重制游戏并返回观察结果
    def reset(self):
        # 重置游戏状态
        self.game.reset()

        # 标记游戏未完成
        self.done = False
        # 奖励步骤计数器清零
        self.reward_step_counter = 0

        # 生成观察结果
        obs = self._generate_observation()
        return obs
    
    #执行游戏的一步操作并进行判断游戏是否结束，根据状态进行奖励和处罚
    def step(self, action):
        # 执行游戏的一步操作，并获取游戏状态和相关信息
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        reward = 0.0
        self.reward_step_counter += 1

        # 判断是否达到步数限制，游戏结束
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        # 如果游戏结束（蛇撞到墙壁或自己）
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # 基于蛇的长度计算游戏结束惩罚
            # # Game Over penalty is based on snake size.
            # # reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)
            # # return obs, reward * 0.1, self.done, info

            # 线性惩罚衰减
            # Linear penalty decay.
            reward = info["snake_size"] - self.grid_size # (-max_growth, 0)
            return obs, reward * 0.1, self.done, info

        # 如果吃到食物
        elif info["food_obtained"]: # food eaten
            # 根据获取食物所需的步数计算奖励
            # Reward on num_steps between getting food.
            reward = math.exp((self.grid_size - self.reward_step_counter) / self.grid_size) # (0, e)
            self.reward_step_counter = 0 # Reset reward step counter

        # 其他情况
        else:
            # 如果蛇头离食物的距离比之前近
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                # 奖励与蛇的长度成反比
                reward = 1 / info["snake_size"] # No upper limit might enable the agent to master shorter scenario faster and more firmly.
            else:
                # 否则，奖励为负数，与蛇的长度成反比
                reward = - 1 / info["snake_size"]

            # # 打印奖励的十分之一
            # print(reward*0.1)
            # # 暂停一秒钟
            # time.sleep(1)

        # 注释：最大分数为 144e - 1 = 390，最小分数为 -141
        # max_score: 144e - 1 = 390
        # min_score: -141 

        # 注释：线性奖励范围
        # Linear:
        # max_score: 288
        # min_score: -141

        # 对奖励进行缩放
        reward = reward * 0.1 # Scale reward
        return obs, reward, self.done, info
    
    #渲染游戏
    def render(self):
        # 调用 game 对象的 render 方法进行渲染
        self.game.render()

    #动作有效性检查的结果
    def get_action_mask(self):
        # 创建一个二维numpy数组，其中包含一个列表的列表，每个内部列表包含对动作有效性检查的结果
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    #检查动作有效性以及蛇的尾部处理
    def _check_action_validity(self, action):
        # 当前游戏的方向
        current_direction = self.game.direction
        # 蛇的列表
        snake_list = self.game.snake
        # 蛇的头部位置
        row, col = snake_list[0]

        if action == 0: # UP
            # 如果当前方向是向下，则不能向上移动
            if current_direction == "DOWN":
                return False
            else:
                # 向上移动一行
                row -= 1

        elif action == 1: # LEFT
            # 如果当前方向是向右，则不能向左移动
            if current_direction == "RIGHT":
                return False
            else:
                # 向左移动一列
                col -= 1

        elif action == 2: # RIGHT 
            # 如果当前方向是向左，则不能向右移动
            if current_direction == "LEFT":
                return False
            else:
                # 向右移动一列
                col += 1     

        elif action == 3: # DOWN 
            # 如果当前方向是向上，则不能向下移动
            if current_direction == "UP":
                return False
            else:
                # 向下移动一行
                row += 1

        # 检查蛇是否撞到了自己或墙壁。注意，如果蛇在当前步骤中没有吃到食物，则蛇的尾部将被移除。
        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                # 如果蛇吃到了食物，则不会移除蛇的最后一个单元格
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                # 检查是否越界
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                # 如果蛇没有吃到食物，则会移除蛇的最后一个单元格
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                # 检查是否越界
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        # 如果游戏结束，则返回False
        if game_over:
            return False
        else:
            return True

    # EMPTY: 0; SnakeBODY: 0.5; SnakeHEAD: 1; FOOD: -1;
    #为智能体提供了一个关于游戏当前状态的简化但有用的表示，使得智能体能够根据这个观察来制定策略并学习如何玩游戏
    def _generate_observation(self):
        # 创建一个全零的观测矩阵，大小为游戏棋盘的大小，数据类型为float32
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.float32)

        # 在观测矩阵中，将蛇的身体位置上的值设置为从0.8到0.2的等差数列，表示蛇的身体
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(0.8, 0.2, len(self.game.snake), dtype=np.float32)

        # 将蛇的头部位置上的值设置为1.0，表示蛇的头部
        obs[tuple(self.game.snake[0])] = 1.0

        # 将食物位置上的值设置为-1.0，表示食物
        obs[tuple(self.game.food)] = -1.0

        # 返回观测矩阵
        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
