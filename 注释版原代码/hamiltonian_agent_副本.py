import time
import random

from snake_game_custom_wrapper_cnn import SnakeEnv

FRAME_DELAY = 0.01 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

BOARD_SIZE = 12

def generate_hamiltonian_cycle(board_size):
    # 初始化路径，第一行从左到右遍历
    path = [(0, c) for c in range(board_size)]

    # 遍历除第一行外的每一行
    for i in range(1, board_size):
        # 如果是偶数行，则从第二列开始从左到右遍历
        if i % 2 == 0:
            for j in range(1, board_size):
                path.append((i, j))
        # 如果是奇数行，则从倒数第二列开始从右到左遍历
        else:
            for j in range(board_size - 1, 0, -1):
                path.append((i, j))

    # 从最后一行开始到第一行，只遍历第一列
    for r in range(board_size - 1, 0, -1):
        path.append((r, 0))
    
    return path

def find_next_action(snake_head, next_position):
    # 计算行差
    row_diff = next_position[0] - snake_head[0]
    # 计算列差
    col_diff = next_position[1] - snake_head[1]

    # 如果行差为1且列差为0，表示向下移动
    if row_diff == 1 and col_diff == 0:
        return 3  # DOWN
    # 如果行差为-1且列差为0，表示向上移动
    elif row_diff == -1 and col_diff == 0:
        return 0  # UP
    # 如果行差为0且列差为1，表示向右移动
    elif row_diff == 0 and col_diff == 1:
        return 2  # RIGHT
    # 如果行差为0且列差为-1，表示向左移动
    elif row_diff == 0 and col_diff == -1:
        return 1  # LEFT
    else:
        return -1

def main():
    # 生成随机种子
    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    # 初始化游戏环境
    env = SnakeEnv(silent_mode=False, seed=seed, board_size=BOARD_SIZE)

    # 生成哈密顿回路
    cycle = generate_hamiltonian_cycle(env.game.board_size)
    # 计算回路的长度
    cycle_len = len(cycle)
    # 当前位置的索引
    current_index = 0

    # 总的步数
    num_step = 0
    # 游戏是否结束
    done = False

    # 当游戏未结束时循环
    while not done:
        # 随机选择一个动作
        action = env.action_space.sample()
        # 获取蛇头的位置
        snake_head = env.game.snake[0]
        # 更新当前位置的索引
        current_index = (current_index + 1) % cycle_len

        # 当当前位置不是蛇头时循环
        while cycle[current_index] != snake_head:
            current_index = (current_index + 1) % cycle_len

        # 获取下一个位置
        next_position = cycle[(current_index + 1) % cycle_len]
        # 根据蛇头和下一个位置找到下一个动作
        action = find_next_action(snake_head, next_position)

        # 执行动作，并获取返回值
        _, _, done, _ = env.step(action)
        # 步数加一
        num_step += 1
        # 渲染游戏环境
        env.render()
        # 暂停一段时间
        time.sleep(FRAME_DELAY)

        # 如果游戏结束
        if done:
            # 打印游戏结束信息
            print(f"Game Finished: Score = {env.game.score}, Total steps = {num_step}")
            # 暂停一段时间
            time.sleep(ROUND_DELAY)

if __name__ == "__main__":
    main()
