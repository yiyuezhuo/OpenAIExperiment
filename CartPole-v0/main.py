# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:05:27 2016

@author: yiyuezhuo

It's guided by
https://zhuanlan.zhihu.com/p/21477488?refer=intelligentunit

I implement it by keras backended on theano instead of tensorflow.
Because my fuck windows can't run tersorflow easily.
"""

import gym
import numpy as np 
import random
from collections import deque

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model


ENV_NAME = 'CartPole-v0'
EPISODE  = 1000 # Episode limitation
STEP     = 300   # Step limitatin in an episode
TEST     = 10



def main():
    # initialize OpenAI Gym env and dqn agent
    env   = gym.make(ENV_NAME)
    agent = DQN(env)
    
    for episode in range(EPISODE):
        print("episode {}/{}".format(episode,EPISODE))
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            #print('step {}/{}'.format(step,STEP))
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            
        # Test every 10 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print("episode: ",episode,"Evaluation Average Reward:", ave_reward)
            if ave_reward >= 200:
                break
    return agent
    
def test_render(agent):
    total_reward = 0
    env = gym.make(ENV_NAME)
    for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
            env.render()
            action = agent.action(state) # direct action for test
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    env.close()
    
def test_monitor_radnom():
    env = gym.make('CartPole-v0')
    env.monitor.start('/tmp/cartpole-experiment-1',force=True)
    for i_episode in range(20):
        total_reward = 0
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("reward {}".format(total_reward))
                break

    env.monitor.close()
    
def test_monitor_agent(agent):
    env = gym.make('CartPole-v0')
    env.monitor.start('/tmp/cartpole-experiment-1',force=True)
    for i_episode in range(100):
        state = env.reset()
        total_reward = 0
        for t in range(200):
            env.render()
            #print(state)
            #action = env.action_space.sample()
            action = agent.action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("reward {}".format(total_reward))
                break
        print("Episode finished after {} timesteps".format(t+1))
        print("reward {}".format(total_reward))
    env.monitor.close()



# Hyper Parameters for DQN
GAMMA = 0.9 # discount Factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
            
class DQN(object):
    # DQN Agent
    def __init__(self, env, model = None): #
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        if model == None:
            self.create_Q_network()
        elif type(model) == str:
            self.model = load_model(model)
        else:
            self.model = model
        #self.create_training_method()
        
    def create_Q_network(self, verbose = True): # 
        if verbose:
            print("start create Q network")
        
        model = Sequential()
        model.add(Dense(20, input_shape=(self.state_dim,)))
        model.add(Activation('relu'))
        model.add(Dense(self.action_dim))
        model.compile(loss='mse',optimizer = Adam())
        
        self.model = model
        
        if verbose:
            print("end create Q network")

        
    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
    def train_Q_network(self, nb_epoch = 100, verbose = False):
        self.time_step += 1
        
        if verbose:
            print('start train network {}'.format(self.time_step))
        
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        
        # Step 2: calculate y
        y_batch = self.model.predict(np.array(state_batch))
        next_batch = self.model.predict(np.array(next_state_batch))
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            a = action_batch[i].tolist().index(1.0)
            if done:
                y_batch[i,a] = reward_batch[i]
            else:
                y_batch[i,a] = reward_batch[i] + GAMMA * np.max(next_batch[i])
        
        self.model.fit(np.array(state_batch), y_batch,
                       #nb_epoch = nb_epoch,
                       #batch_size = len(state_batch),
                       verbose = 0)
        
    def egreedy_action(self, state):
        state = np.array([state])
        Q_value = self.model.predict(state)
        if random.random() <= self.epsilon:
            return np.random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)
    def action(self, state):
        state = np.array([state])
        return np.argmax(self.model.predict(state))
        
class RandomWalker(object):
    def __init__(self):
        pass
    def action(self,_):
        if random.random() > 0.5:
            return 1
        else:
            return 0
            
env = gym.make(ENV_NAME)
agent = DQN(env, model = 'mymodel.h5')