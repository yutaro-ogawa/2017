# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

# -- constants
ENV = 'CartPole-v0'
NUM_STATES = gym.make(ENV).observation_space.shape[0]     # CartPoleは4状態
NUM_ACTIONS = gym.make(ENV).action_space.n        # CartPoleは、右に左に押す2アクション
NONE_STATE = np.zeros(NUM_STATES)


RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
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


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        #self.default_graph.finalize()  # avoid modifications
        self.weights = self.model.get_weights()  # ネットワークの重みを保存

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:   # optimizerから同時に呼び出されないように、排他ロックする

            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

            s = np.vstack(s)
            a = np.vstack(a)
            r = np.vstack(r)
            s_ = np.vstack(s_)
            s_mask = np.vstack(s_mask)

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
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s, weights):
        with self.default_graph.as_default():
            self.model.set_weights(weights)
            p, v = self.model.predict(s)
            return p

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0


class Agent:
    def __init__(self, agent_brain):
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_steps = EPS_STEPS

        self.memory = []  # used for n_step return
        self.R = 0.

        self.agent_weights = agent_brain.weights


    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            #p = brain.predict_p(s,brain.weights)[0]
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# --実際に行動して、経験を積むスレッドクラスです　-------
class Environment(threading.Thread):    # threadを継承しています
    stop_signal = False     # 終了命令のフラグ
    total_reward_vec = np.zeros(5)   # 総報酬を10試行分格納して、平均総報酬をもとめる
    total_trial_each_thread = 0     # 各スレッドの試行数

    def __init__(self, thread_name, thread_brain, render):
        threading.Thread.__init__(self, name=thread_name)
        self.render = render
        self.env = gym.make(ENV)
        self.thread_brain = thread_brain
        self.agent = Agent(self.thread_brain)

    def runEpisode(self):

        s = self.env.reset()
        R = 0

        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], R))     # 0個目を破棄して新しい10個に
                self.total_trial_each_thread += 1   # このスレッドの総試行回数を増やす
                break

        # 総試行数、スレッド名、今回の報酬を出力
        # print(self.name+":今回の報酬R:"+str(R))

    def run(self):
        while not self.stop_signal:
            self.runEpisode()   # ずっとrunEpisode()を繰り返す

    def stop(self):
        self.stop_signal = True


# --brainに最適化を指令するスレッドクラスです　-------
class Optimizer(threading.Thread):  # pythonのthreadクラスを継承しています
    stop_signal = False

    def __init__(self, thread_name):
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        while not self.stop_signal:
            brain.optimize()    # ずっとbrainを最適化し続ける

    def stop(self):
        self.stop_signal = True


# -- main ここからメイン関数です。
total_trial = 0         # 総試行数
averaged_reward = 0     # 全スレッドの平均の総報酬

brain = Brain()  # すべてのスレッドで共有する学習するニューラルネットワークのクラスです

env_test = Environment(thread_name="学習後スレッド", thread_brain=brain, render=True)   # 学習終了後のCartPoleを描画するのに使います
envs = [Environment(thread_name="Envスレッド"+str(i+1), thread_brain=brain, render=False) for i in range(THREADS)]      # スレッドクラスを継承したもの複数生成
opts = [Optimizer(thread_name="Optスレッド"+str(i+1)) for i in range(OPTIMIZERS)]     # スレッドクラスを継承したもの複数生成

for o in opts:
    o.start()   # Threadのstar()関数です。その後run()が呼び出されます。run()はstop()が呼ばれるまで、何度もrunEpisode()を繰り返します

for e in envs:
    e.start()   # Threadのstar()関数です。その後run()が呼び出されます。run()はstop()が呼ばれるまで、続きます


while True:     # 学習が終わるまで回し続けます
    time.sleep(1.0)  # 10秒の間、メインスレッドの処理を止めて、各スレッドを動かします
    for e in envs:  # 全スレッドの学習度をチェックします
        total_trial += e.total_trial_each_thread        # 各スレッドの試行数を足す
        averaged_reward += e.total_reward_vec.mean()    # 各スレッドの平均総報酬を足す

    averaged_reward = averaged_reward / THREADS
    print("総試行数：" + str(total_trial) + ":平均の総報酬：" + str(averaged_reward))
    if averaged_reward > 195:  # 全スレッドの平均が195を越えたら学習終了
        break
    else:
        total_trial = 0
        averaged_reward = 0


for e in envs:
    e.stop()    # 全スレッドでstop_signalをTrueにします
for e in envs:
    e.join()    # 各スレッドが終了するまで待機します

for o in opts:
    o.stop()    # 全スレッドでstop_signalをTrueにします
for o in opts:
    o.join()    # 各スレッドが終了するまで待機します

print("学習終了です。"+"総試行数：" + str(total_trial)+":学習の挙動を実行します")
#env_test.run()  # 学習後の挙動を表示する
