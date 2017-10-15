# coding:utf-8

# OpenGym CartPole-v0 with A3C on CPU
# -----------------------------------
#
# A3C implementation with TensolFlow multi threads.
#
# Made as part of Qiita article, available at
# https://??/
#
# author: Sugulu, 2017

import numpy as np
import tensorflow as tf
import gym, time, random, threading
from keras.models import *
from keras.layers import *
from keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensolFlow高速化用のワーニングを表示させない

# -- constants of Game
ENV = 'CartPole-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]     # CartPoleは4状態
NUM_ACTIONS = env.action_space.n        # CartPoleは、右に左に押す2アクション
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of TensolFlow multi threads
N_WORKERS = 8
SERVER_THREAD_NAME = "パラメーターサーバースレッド"



RUN_TIME = 30
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_END = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient



# --TensolFlowのDeep Neural Networkのクラスです　-------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask

    def __init__(self):
        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.weights = self.model.get_weights()  # ネットワークの重みを保存

    # 関数名がアンダースコア2つから始まるものは「外部から参照されない関数」、「1つは基本的に参照しない関数」という意味
    def _build_model(self):     # Kerasでネットワークの形を定義します
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(model):      # TensolFlowでネットワークの重みをどう学習させるのかを定義します
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))  # placeholderは変数が格納される予定地となります
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        # loss関数を定義します
        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        # loss関数を最小化していくoptimizerの定義です
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):     # ネットワークの重みを学習・更新します
        if len(self.train_queue[0]) < MIN_BATCH:    # データがたまっていない場合は更新しない
            time.sleep(0)  # yield
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)    # vstackはvertical-stackで縦方向に行列を連結、いまはただのベクトル転置操作
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        # 経験の増加に対して、学習が追いついていない時はアラートします
        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        s_t, a_t, r_t, minimize = self.graph

        #self.weights = self.model.get_weights()
        #self.model.set_weights(self.weights)
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

        #self.weights = self.model.get_weights()

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)


    def predict(self, s):
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
            p, v = self.model.predict(s)
            return v


# --行動を決定するクラスです、CartPoleであれば、棒付き台車そのものになります　-------
class Agent:
    def __init__(self):
        self.brain = Brain()    # 行動を決定するための脳（ニューラルネットワーク）
        self.memory = []        # s,a,r,s_の保存メモリ、　used for n_step return
        self.R = 0.             # 時間割引した、「いまからNステップ分あとまで」の総報酬R

        self.agent_weights = self.brain.weights

    def act(self, s):   # ε-greedy法で行動を決定します

        # 全スレッドトータルの行動回数をひとつ増やします
        global frames  # global変数を書き換える場合は、関数内でglobal宣言が必要です
        frames = frames + 1

        if frames >= EPS_END:
            eps = EPS_END
        else:
            eps = EPS_START + frames * (EPS_END - EPS_START) / EPS_STEPS  # linearly interpolate

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)   # ランダムに行動
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]      # 状態sから行動確率pi=[右がよい確率 , 左がよい確率]をもとめる
            # a = np.argmax(p)  # これだと確率最大の行動を、毎回選択
            a = np.random.choice(NUM_ACTIONS, p=p)  # probability = p のこっちだと、確率pにしたがって、行動を選択
            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect



# --スレッドになるクラスです　-------
class Worker:
    def __init__(self, thread_name, render):
        self.name = thread_name

    def run(self):
        global var

        if self.name != SERVER_THREAD_NAME:
            var = var + self.name

        print(self.name+var+str(frames))


# -- main ここからメイン関数です------------------------------
# M0.変数の定義と、セッションの開始です
total_trial = 0         # 総試行数
averaged_reward = 0     # 全スレッドの平均の総報酬

SESS = tf.Session()
K.set_session(SESS)
K.manual_variable_initialization(True)

# 全スレッドで共有して使用するグローバル変数を定義します
frames = 1
var = "aaa"


# M1.スレッドを作成します
with tf.device("/cpu:0"):
    threads= []     # 並列して走るスレッド
    # Serverとなるスレッド
    thread_name = SERVER_THREAD_NAME
    threads.append(Worker(thread_name, render=False))

    # 経験を積むスレッド
    for i in range(N_WORKERS):
        thread_name = "Envスレッド"+str(i+1)
        threads.append(Worker(thread_name, render=False))

# M2.TensolFlowでマルチスレッドを実行します
COORD = tf.train.Coordinator()                  # TensolFlowでマルチスレッドにするための準備です
SESS.run(tf.global_variables_initializer())     # TensolFlowを使う場合、最初に変数初期化をして、実行します

running_threads = []
for worker in threads:
    t = threading.Thread(target=worker.run())
    t.start()
    running_threads.append(t)

# M3.スレッドの終了を合わせます
COORD.join(running_threads)

