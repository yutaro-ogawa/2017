# coding:utf-8
import gym  # 倒立振子(cartpole)の実行環境
from gym import wrappers  # gymの画像保存
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class QNetwork:  # [A]Neural Networkを定義
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        # state inputs to the Q-network
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)


class Memory:  # [B]状態と行動を保存するバッファーメモリを定義
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]


class Actor:
    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

        else:
            action = np.random.choice([0, 1])  # ランダムに行動する

        return action


# [C] メイン関数開始----------------------------------------------------
# [D] 初期設定--------------------------------------------------------
env = gym.make('CartPole-v0')
max_number_of_steps = 200  # 1試行のstep数
num_consecutive_iterations = 10  # 学習完了評価に使用する平均試行回数
num_episodes = 1000  # 総試行回数
goal_average_reward = 195  # この報酬を超えると学習終了（中心への制御なし）
# goal_average_reward = 150  #【中央に留める用】（中心への制御あり）
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
final_x = np.zeros((num_episodes, 1))  # 学習後、各試行のt=200でのｘの位置を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
isrender = 0  # 描画フラグ
# ---
hidden_size = 16               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.001         # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
batch_size = 64                # Q-networkを更新するバッチの大記載
pretrain_length = batch_size   # 開始前に事前に学習してゔバッファーメモリに入れておく量

# [E] ネットワークとメモリの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)
memory = Memory(max_size=memory_size)
actor = Actor()

# [F] メモリに初期の状況を保存--------------------------------------------------------
env.reset()  # cartPoleの環境初期化
state, reward, done, _ = env.step(env.action_space.sample())    # 適当な行動をとる
state = np.reshape(state, [1, 4])


# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulationif done:  # The simulation fails so no next state
    # env.render()    next_state = np.zeros(state.shape)
    # Make a random action
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    next_state = np.reshape(observation, [1, 4])

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        observation, reward, done, _ = env.step(env.action_space.sample())
        state = np.reshape(observation, [1, 4])
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

#############
# Training
#############
# E. メインメソッド--------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    env.reset()  # cartPoleの環境初期化
    state, reward, done, _ = env.step(env.action_space.sample())  # 適当な行動をとる
    state = np.reshape(state, [1, 4])
    episode_reward = 0

    for t in range(max_number_of_steps + 1):  # 1試行のループ
        if islearned == 1:  # 学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print(state[0, 0])  # カートのx位置を出力
        action = actor.get_action(state, episode, mainQN)   # tでの行動を決定する
        next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
        next_state = np.reshape(next_state, [1, 4])

        # 中央にとどめる報酬制御
        if state[0, 0] > 0.3 or state[0, 0] < -0.3:
            done = 1

        # 報酬を設定し、与える
        if done:
            next_state = np.zeros(state.shape)  # 次の状態はない
            if t < 195:
                reward = -1  # クリッピングで報酬は1, 0, -1に固定
            else:
                reward = 1  # 立ったまま終了時は報酬
        else:
            reward = 1  # 各ステップで立ってたら報酬追加　はじめからrewardに1が入っているが、明示的に表す

        episode_reward += reward  # 報酬を追加

        memory.add((state, action, reward, next_state)) # メモリと状態の更新する
        state = next_state

        if done:    # 終了時の処理
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))

            if islearned == 1:  # 学習終わってたら最終のx座標を格納
                final_x[episode, 0] = state[0, 0]
            break

        # Qネットワークの学習 小川ここを別クラスにしたい、DDQNにもしたい
        if np.mod(t, 2) == 0:
            inputs = np.zeros((batch_size, 4))
            targets = np.zeros((batch_size, 2))
            mini_batch = memory.sample(batch_size)

            for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
                inputs[i:i + 1] = state_b
                target = reward_b

                if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                    target_Q = mainQN.model.predict(next_state_b)[0]
                    target = reward_b + gamma * np.amax(mainQN.model.predict(next_state_b)[0])
                targets[i] = mainQN.model.predict(state_b)
                targets[i][action_b] = target
            mainQN.model.fit(inputs, targets, epochs=1, verbose=0)  #epochsは訓練データの反復回数、verbose=0は表示なしの設定

    #複数施行の平均報酬で終了を判断
    if (total_reward_vec.mean() >= goal_average_reward):  # 直近の平均エピソードが規定報酬以上であれば成功
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        # np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #Qtableの保存する場合
        if isrender == 0:
            # env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            isrender = 1
            # 10エピソードだけでどんな挙動になるのか見たかったら、以下のコメントを外す
            # if episode>10:
            #    if isrender == 0:
            #        env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            #        isrender = 1
            #    islearned=1;

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")

