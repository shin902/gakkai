import numpy as np
import random

# 迷路の設定
maze = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
]

# 初期設定
goal_pos = [3, 3]
start_pos = [0, 0]

# 移動の定義
def get_next_position(pos, action):
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上、右、下、左
    new_pos = [
        pos[0] + moves[action][0],
        pos[1] + moves[action][1]
    ]
    return new_pos

# 行動が有効かチェック
def is_valid_move(pos):
    return (0 <= pos[0] < len(maze) and
            0 <= pos[1] < len(maze[0]) and
            maze[pos[0]][pos[1]] != 1)

# 報酬関数
def get_reward(pos):
    if pos == goal_pos:
        return 100
    elif not is_valid_move(pos):
        return -10
    return -1

# 状態の数値化
def state_to_number(pos):
    return pos[0] * 4 + pos[1]

# Q学習関数
def q_learning(num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    # Q-テーブルの初期化（状態数16、行動数4）
    q_table = np.zeros((16, 4))

    for _ in range(num_episodes):
        # エピソードの開始
        current_pos = start_pos.copy()

        while current_pos != goal_pos:
            # 現在の状態
            current_state = state_to_number(current_pos)

            # 行動選択（ε-greedy戦略）
            if random.random() < epsilon:
                action = random.randint(0, 3)  # ランダムな行動
            else:
                action = np.argmax(q_table[current_state])

            # 次の位置を計算
            next_pos = get_next_position(current_pos, action)

            # 移動が有効な場合のみ実行
            if is_valid_move(next_pos):
                # 次の状態
                next_state = state_to_number(next_pos)

                # 報酬の取得
                reward = get_reward(next_pos)

                # Q値の更新
                best_next_action = np.argmax(q_table[next_state])
                td_target = reward + discount_factor * q_table[next_state][best_next_action]
                td_error = td_target - q_table[current_state][action]
                q_table[current_state][action] += learning_rate * td_error

                # 位置の更新
                current_pos = next_pos

    return q_table

# トレーニングの実行
trained_q_table = q_learning()

# 学習結果の可視化
def visualize_path():
    current_pos = start_pos.copy()
    path = [current_pos.copy()]

    while current_pos != goal_pos:
        current_state = state_to_number(current_pos)
        best_action = np.argmax(trained_q_table[current_state])

        next_pos = get_next_position(current_pos, best_action)

        if is_valid_move(next_pos):
            current_pos = next_pos
            path.append(current_pos.copy())
        else:
            break

    return path

# 最適経路の表示
print("学習した最適経路:")
optimal_path = visualize_path()
for pos in optimal_path:
    print(pos)
