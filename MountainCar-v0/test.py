# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:07:06 2017

@author: 8233900
"""

import gym

ENV_NAME = 'MountainCar-v0'
TEST = 200

env   = gym.make(ENV_NAME)
env.reset()
for i in range(TEST):
    env.render()
    a = env.action_space.sample()
    env.step(a)
env.close()

env = gym.make(ENV_NAME)
env.reset()
for i in range(TEST):
    env.render()
    #a = env.action_space.sample()
    env.step(0)
env.close()

env = gym.make(ENV_NAME)
env.reset()
for i in range(TEST):
    env.render()
    #a = env.action_space.sample()
    env.step(1)
env.close()

env = gym.make(ENV_NAME)
env.reset()
for i in range(TEST):
    env.render()
    #a = env.action_space.sample()
    env.step(2)
env.close()