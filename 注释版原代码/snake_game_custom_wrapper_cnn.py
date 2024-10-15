import math

import gym
import numpy as np

from snake_game import SnakeGame
#   math：提供数学函数。
#	gym：OpenAI Gym库，用于创建和使用强化学习环境。
#	numpy：用于数组操作的科学计算库。
#	SnakeGame：自定义的贪吃蛇游戏逻辑类，从snake_game`模块导入。

class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        #   seed = 0：随机种子，用于确定随机数生成器的初始状态，以便每次运行程序时得到相同的随机结果。
        #  board_size = 12：棋盘的大小，表示棋盘是一个12x12的方格。
        #   silent_mode = True：静音模式，控制是否在游戏过程中播放声音。
        #   limit_step = True：是否限制步数，用于决定游戏的最大步数。
        super().__init__()
        # 调用父类gym.Env的构造函数，确保正确初始化环境
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()
       # 创建SnakeGame实例，并传递随机种子、棋盘大小和静音模式参数。
       # 调用self.game.reset()来重置游戏到初始状态
        self.silent_mode = silent_mode
        #将静音模式参数存储到实例变量self.silent_mode中。
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        #定义动作空间为4个离散动作，分别对应向上、向左、向右和向下移动。
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        #定义观察空间为形状为(84, 84, 3)的三通道图像，像素值范围在0到255之间，数据类型为uint8。
        #见注释1
        self.board_size = board_size
        self.grid_size = board_size ** 2
        #将棋盘大小存储到实例变量self.board_size中。
        #计算棋盘的总格子数并存储到实例变量self.grid_size中，这是蛇的最大长度。
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size
        #获取初始蛇的长度并存储到实例变量self.init_snake_size中。
        #计算蛇的最大增长量并存储到实例变量self.max_growth中，这是总格子数减去初始蛇长度。
        self.done = False
        #初始化游戏结束标志为False
        if limit_step:
            self.step_limit = self.grid_size * 4
        else:
            self.step_limit = 1e9 # Basically no limit.
        #如果启用了步数限制，将步数限制设置为grid_size的4倍，确保有足够的步数来获得食物。
        #如果未启用步数限制，将步数限制设置为非常大的值1e9，实际上没有限制。
        self.reward_step_counter = 0
        #初始化奖励步数计数器为0。

    def reset(self):
        self.game.reset()
        #调用self.game.reset()重置贪吃蛇游戏的状态，包括重置蛇的位置、方向、食物的位置等
        self.done = False
        #将游戏结束标志self.done设置为False，表示新的一局游戏还没有结束。
        self.reward_step_counter = 0
        #将奖励步数计数器self.reward_step_counter重置为0，用于记录当前游戏中步数的计数器。
        obs = self._generate_observation()
        #调用self._generate_observation()生成当前游戏状态的观察结果。这个函数会返回一个表示当前游戏状态的图像数组。
        return obs

   # reset函数：重置游戏状态，重置游戏结束标志和步数计数器，并生成初始观察图像返回。

    def step(self, action):
        self.done, info = self.game.step(action)
        # 调用self.game.step(action)来执行给定的动作action。返回两个值：
        # self.done：布尔值，表示游戏是否结束。
        # info：字典，包含游戏的相关信息，如蛇的大小、蛇头的位置、食物的位置等。
        obs = self._generate_observation()
        #调用self._generate_observation() 生成当前游戏状态的观察结果。
        reward = 0.0
        self.reward_step_counter += 1
        #初始化奖励值reward为0.0。
       # 增加奖励步数计数器self.reward_step_counter。

        #以下为胜利条件检测
        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        #如果蛇的大小等于棋盘的总格子数（即蛇填满了整个棋盘），则：
        #设置奖励为self.max_growth * 0.1。
        #设置游戏结束标志self.done为True。
        #如果不是静音模式，播放胜利的声音。
        #返回观察值、奖励、游戏结束标志和信息。
        
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
        #如果奖励步数计数器超过了步数限制，则：
        #将奖励步数计数器重置为0。
        #设置游戏结束标志self.done为True。
        
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            reward = reward * 0.1
            return obs, reward, self.done, info
        #如果游戏结束，则：根据蛇的大小计算惩罚奖励。使用math.pow函数计算惩罚值，奖励在 - self.max_growth到 - 1之间，乘以0.1。
        #返回观察值、奖励、游戏结束标志和信息。
          
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0 # Reset reward step counter
            #如果蛇吃到了食物，则：设置奖励为蛇的大小除以棋盘的总格子数。
           #将奖励步数计数器重置为0。
        
        else:

            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1
        #否则，根据蛇头位置与食物位置的距离变化给出微小的奖励或惩罚：
        #如果蛇头接近食物，给予微小奖励。如果蛇头远离食物，给予微小惩罚。奖励值乘以0.1进行缩放。
        #奖励步数计步器见注释2

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward, self.done, info
        #返回新的观察值obs、奖励reward、游戏结束标志self.done和信息info。

    def render(self):
        self.game.render()
        #render函数：调用SnakeGame类的渲染方法显示游戏画面。

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    #见注释3

    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
       # 获取当前蛇的方向（current_direction）和蛇的身体位置列表（snake_list）。
       # 获取蛇头的当前位置（row, col）。
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1
       # 如果动作是向上（action == 0）：检查当前方向是否是向下（蛇不能立即反向移动）。
        #如果是向下，返回False，表示该动作无效。否则，将row减1，模拟向上移动。
        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1
       # 如果动作是向左（action == 1）：检查当前方向是否是向右（蛇不能立即反向移动）。
       # 如果是向右，返回False，表示该动作无效。否则，将col减1，模拟向左移动。

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
            #检查移动后的位置是否是食物位置。
            #如果是食物位置，则检查移动后的新位置是否会导致游戏结束。
            #游戏结束的条件包括：新位置在蛇身体上。 新位置超出棋盘边界
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
       # 检查移动后的位置是否是食物位置。如果是食物位置，则检查移动后的新位置是否会导致游戏结束。

        if game_over:
            return False
        else:
            return True
       # 如果新位置会导致游戏结束，返回False，表示该动作无效。 否则，返回True，表示该动作有效。

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)
        #创建一个self.game.board_size x self.game.board_size的二维数组，初始化为0。这个数组用来表示游戏棋盘，每个元素表示一个像素。

        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
       # 将蛇的身体部分设为灰色，颜色强度从200逐渐减弱到50。通过np.transpose(self.game.snake) 得到蛇身体的坐标，然后使用np.linspace生成颜色强度的线性渐变数组。

        obs = np.stack((obs, obs, obs), axis=-1)
        #将二维数组扩展为三通道图像，形成一个(board_size, board_size, 3)的三维数组。每个通道的值都相同，初始情况下所有通道都是灰色。

        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]
        #将蛇的头部设为绿色[0, 255, 0]，将蛇的尾部设为红色[255, 0, 0]
        obs[self.game.food] = [0, 0, 255]
        #将食物的位置设为蓝色[0, 0, 255]。

        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)
        #将生成的图像放大7倍，使其变为(84, 84, 3)的图像，便于显示和处理。
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
