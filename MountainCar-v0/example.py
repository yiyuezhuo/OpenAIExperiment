# -*- coding: utf-8 -*-
"""
https://gym.openai.com/evaluations/eval_89nQ59Y4SbmrlQ0P9pufiA
"""

import theano
from theano import tensor as T
import lasagne
import numpy as np
from collections import deque
import copy
import gym


class ValueFunctionDQN:
    def __init__(self, model, learning_rate=1e-3):
        states = T.matrix('states', dtype='float64')
        targets = T.matrix('targets', dtype='float64')
        #actions = T.vector('rewards',dtype='int64')
        
        action_values = lasagne.layers.get_output(model, states)
        
        weights = lasagne.layers.get_all_params(model, trainable=True)
        
        loss = T.mean((targets - action_values)**2)
        updates = lasagne.updates.adam(loss, weights, learning_rate=learning_rate)
        #updates = lasagne.updates.nesterov_momentum(loss, weights, learning_rate=learning_rate, momentum=0.9)
        #updates = lasagne.updates.sgd(loss, weights, learning_rate=learning_rate)
        #updates = lasagne.updates.rmsprop(loss, weights, learning_rate=learning_rate)
        
        # test values
        max_weight = T.max([T.max(w) for w in weights])

        self.train_func = theano.function([states, targets], 
                                     [loss, max_weight], 
                                     updates=updates)

        self.q_func = theano.function([states], action_values)

        #self.test_av = theano.function([states, actions], action_values[T.arange(actions.shape[0]),actions])
        
    def predict(self, states):
        return self.q_func(states)
        
    def train(self, states, targets):
        return self.train_func(states, targets)


class AgentEpsGreedy:
    def __init__(self, n_actions, model, eps=1., learning_rate=0.001):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.value_func_dqn = ValueFunctionDQN(model, self.lr)
        self.eps = eps
        
    def act(self, state):
        action_values = self.value_func_dqn.predict([state])[0]
        
        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps
        
        return np.random.choice(self.n_actions, p=policy) 
    
    def train(self, states, targets):
        return self.value_func_dqn.train(states, targets)
    
    def predict(self, states):
        return self.value_func_dqn.predict(states)


class ReplayMemory:
    def __init__(self, max_size=128):
        self.memory = deque(maxlen=max_size)
    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        idxs = np.random.choice(len(self.memory), batch_size)
        return [self.memory[idx] for idx in idxs]
        
    def add(self, item):
        self.memory.append(item)


def run_episode(env, 
                agent, 
                state_normalizer, 
                memory,
                batch_size,
                discount,
                max_step=10000):
    state = env.reset()
    if state_normalizer is not None:
        state = state_normalizer.transform(state)[0]
    done = False
    for i in range(max_step):
        if done:
            break
        action = agent.act(state)
        state_next, reward, done, info = env.step(action)
        #if i % 1000 == 0:
            #print('Step {}, a={}, s={}'.format(i, action, state_next))
            #if len(memory.memory) > 128:
            #    print('\t{}'.format(targets_b[0]))
        if state_normalizer is not None:
            state_next = state_normalizer.transform(state_next)[0]
        memory.add((state, action, reward, state_next, done))
        
        if len(memory.memory) > batch_size:
            states_b, actions_b, rewards_b, states_n_b, done_b = zip(*memory.sample(batch_size))
            states_b = np.array(states_b)
            actions_b = np.array(actions_b)
            rewards_b = np.array(rewards_b)
            states_n_b = np.array(states_n_b)
            done_b = np.array(done_b).astype(int)
            targets_b = rewards_b + (1.-done_b) * discount * agent.predict(states_n_b).max(axis=1)
            targets = agent.predict(states_b)
            for j, action in enumerate(actions_b):
                targets[j, action] = targets_b[j]
            loss, max_w = agent.train(states_b, targets)
        state = copy.copy(state_next)
    print('{}, w={}, Step={}, a={}, q={}'.format(loss, max_w, i, action, agent.predict([[0,0]])))
    return loss, max_w     
    

def build_model(state_dim, n_actions):
    nn = lasagne.layers.InputLayer(shape=(None,state_dim))
    nn = lasagne.layers.DenseLayer(nn, num_units=512, 
                                   #W=lasagne.init.GlorotNormal(),
                                   #b=lasagne.init.Constant(0.),
                                   nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(nn, num_units=256, 
                                   #W=lasagne.init.GlorotNormal(),
                                   #b=lasagne.init.Constant(0.),
                                   nonlinearity=lasagne.nonlinearities.rectify)
    '''nn = lasagne.layers.DenseLayer(nn, num_units=16, 
                                   #W=lasagne.init.GlorotNormal(),
                                   #b=lasagne.init.Constant(0.),
                                   nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(nn, num_units=50, 
                                   W=lasagne.init.GlorotNormal(),
                                   b=lasagne.init.Constant(0.),
                                   nonlinearity=lasagne.nonlinearities.rectify)
    nn = lasagne.layers.DenseLayer(nn, num_units=30, 
                                   W=lasagne.init.GlorotNormal(),
                                   b=lasagne.init.Constant(0.),
                                   nonlinearity=lasagne.nonlinearities.rectify)'''
    nn = lasagne.layers.DenseLayer(nn, num_units=n_actions, 
                                   #W=lasagne.init.Normal(), 
                                   #b=Constant(0), 
                                   nonlinearity=lasagne.nonlinearities.identity, name = "output_dense_layer")
    return nn           


if __name__ == "__main__":
    nn = build_model(n_actions=3, state_dim=2)
    memory = ReplayMemory(max_size=100000)    
    agent = AgentEpsGreedy(model=nn, n_actions=3, eps=0.9, learning_rate=0.0001)
    env = gym.make("MountainCar-v0")    

    num_episodes = 900
    discount = 0.99#9
    decay_eps = 0.9
    batch_size = 64

    for ep in range(num_episodes):
        loss, max_w = run_episode(env, 
                                  agent, 
                                  None,#state_nrlzr, 
                                  memory,
                                  batch_size = batch_size,
                                  discount=discount,
                                  max_step=15000)

        if agent.eps > 0.0001:
            agent.eps *= decay_eps

    # Result
    env = gym.make("MountainCar-v0")
    env.monitor.start(args.r, force=True)
    av_reward = []
    for ep in range(args.e):
        total_reward = 0
        state = env.reset()
        for step in range(args.s):
            #state = state_nrlzr.transform(state)
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                av_reward.append(total_reward)
                #print "Episode {}: reward {}".format(ep, reward)
                break
    print(np.mean(av_reward))
    env.monitor.close()
    gym.upload('MY_NAME', api_key='#####', ignore_open_monitors=True)
