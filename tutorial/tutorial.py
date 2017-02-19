# -*- coding: utf-8 -*-
import gym

env = gym.make('CartPole-v0')

for i_episode in range(20):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
