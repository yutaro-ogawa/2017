# coding:utf-8
import gym  #倒立振子(cartpole)の実行環境
from gym import wrappers  #gymの画像保存
import numpy as np
import time

# 初期設定--------------------------------------------------------
env = gym.make('CartPole-v0')

goal_average_reward = 150  #150 or 195 #この報酬を超えると学習終了
max_number_of_steps = 200  #1試行のstep数
num_consecutive_iterations = 100  #学習完了評価に使用する平均試行回数
num_episodes = 1500  #総試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)
final_x = np.zeros((num_episodes, 1))  #学習後、各試行のt=200でのｘの位置
islearned = 0  #学習が終わったフラグ
isrender = 0  #描画

#状態を6分割^（4変数）にデジタル変換-----------------------------------------
num_dizitized = 6
#分割数
#Qテーブルを定義
q_table = np.random.uniform(
    low=-1, high=1, size=(num_dizitized**4, env.action_space.n))


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    # 各値を離散値に変換
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    # 状態を変数に変換に変換
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


#次の行動の決定とQテーブルの更新関数--------------------------------------
def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)

    #徐々に最適行動のみをとる、ε-greedy法
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])

    # Qテーブルの更新
    if islearned == 0:
        gamma = 0.95
        alpha = 0.2
        q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state


#メインメソッド開始--------------------------------------------------
for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0
    for t in range(max_number_of_steps):

        # CartPoleの描画（学習終了したら）
        if islearned == 1:
            env.render()
            time.sleep(0.1)
            print [observation[0]]  #カートのx位置

        # 行動の実行とフィードバックの取得
        observation, reward, done, info = env.step(action)

        # 報酬を与える
        if done:
            if t < 195:
                reward = -200  #こけたら罰則
            else:
                reware = 0  #立ったまま終了時は罰則はなし
        else:
            reward = 1  #各ステップで立ってたら報酬追加

        episode_reward += reward

        #中央にとどめる報酬
        if observation[0] > 0.3 or observation[0] < -0.3:
            reward = -10
            episode_reward += reward

        # 次の行動の選択
        action, state = get_action(state, action, observation, reward, episode)

        #終了時の処理
        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:],
                                          episode_reward))  #報酬を記録
            if islearned == 1:
                final_x[episode, 0] = observation[0]
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # 直近の100エピソードが規定報酬以上であれば成功
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        #np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #Qtableの保存する場合
        if isrender == 0:
            #env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            isrender = 1

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")
