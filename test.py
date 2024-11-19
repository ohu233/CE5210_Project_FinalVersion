import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import requests
from Crawler import BusCrawler

import matplotlib.pyplot as plt
import numpy as np
from Env import BusEnv
from DQN import DuelingDoubleDQNAgent

def test_trained_model_with_visualization(env, agent, episodes=1):
    """
    测试已训练的模型，在环境中运行并观察结果，
    可视化累积奖励以及泊位状态和排队情况。
    
    参数：
    - env: BusEnv 环境
    - agent: 已训练的 DuelingDoubleDQNAgent
    - episodes: 测试的回合数
    """
    agent.epsilon = 0.0  # 确保测试时行为是确定性的
    rewards_over_time = []  # 用于存储奖励以便可视化
    
    plt.ion()  # 启用交互模式以实时更新图像

    for episode in range(episodes):
        state = env.reset()
        state = np.array(state)
        done = False
        step = 0
        episode_rewards = []  # 记录每一回合的奖励

        # 创建图像和子图，用于实时可视化
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state)
            state = next_state
            episode_rewards.append(reward)  # 累积奖励
            step += 1

            # 绘制累积奖励
            axs[0].cla()  # 清空累积奖励图
            axs[0].plot(range(len(episode_rewards)), episode_rewards, label='reward')
            axs[0].set_title(f"reward per step")
            axs[0].set_xlabel("step")
            axs[0].set_ylabel("reward")
            axs[0].legend()
            axs[0].grid(True)

            # 获取泊位占用和排队长度数据
            bay_occupancy = [
                1 if env.bays[i][j] is not None else 0 
                for i in range(env.bay_num) for j in range(env.capacity)
            ]
            waiting_queues = [len(queue) for queue in env.waiting_queue]

            # 定义每个泊位状态和排队位置，确保 queue2 位置正确
            bay_and_queue_positions = [
                bay_occupancy[0], bay_occupancy[1], waiting_queues[0],  # 泊位 1 的状态和排队
                bay_occupancy[2], bay_occupancy[3], waiting_queues[1]   # 泊位 2 的状态和排队
            ]

            # 绘制泊位和排队状态
            axs[1].cla()  # 清空泊位状态图
            labels = ["bay1-1", "bay1-2", "queue1", "bay2-1", "bay2-2", "queue2"]
            axs[1].bar(labels, bay_and_queue_positions, color=['blue', 'blue', 'orange', 'blue', 'blue', 'orange'])
            axs[1].set_title("Test")
            axs[1].set_ylabel("bay status / queue length")
            axs[1].grid(True)

            # 暂停以刷新图表
            plt.pause(0.1)

    plt.ioff()  # 测试结束后禁用交互模式
    plt.savefig('test.png')


def test_trained_model(env, agent, episodes=10):
    """
    Test the trained model by running it in the environment and observing the results.
    
    Parameters:
    - env: the BusEnv environment
    - agent: the trained DuelingDoubleDQNAgent
    - episodes: number of episodes to test the model
    """
    agent.epsilon = 0.0  # Set epsilon to 0 to ensure the model is fully deterministic

    for episode in range(episodes):
        state = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.act(state)  # Use the trained model to select actions
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state)
            state = next_state
            total_reward += reward
            step += 1
            
            # Print step details for this episode
            print(f"Episode: {episode + 1}, Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
            print("Current bay status:")
            for i in range(env.bay_num):
                bay_status = [veh['ServiceNo'] if veh else None for veh in env.bays[i]]
                print(f"Bay {i+1}: {bay_status}")
                
            print("Current waiting queue:")
            for i in range(env.bay_num):
                queue_status = [veh['ServiceNo'] for veh in env.waiting_queue[i]]
                print(f"Queue {i+1}: {queue_status}")
            print('-' * 100)
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward}\n")

# 获取当前的日期和时间
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
print("当前的日期和时间:", formatted_datetime)

# 设置 API 和采集数据
api_key = "LhPnk7kfTDqb849G9KuhqA=="
urls = [
    "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=04167",
    "https://datamall2.mytransport.sg/ltaodataservice/v3/BusArrival?BusStopCode=04168"
]
collector = BusCrawler(api_key, urls)


data = collector.collect()

# 创建环境和 DQN Agent
env = BusEnv(data)
agent = DuelingDoubleDQNAgent(state_size=env.state_dim, action_size=env.action_space)
# Load the saved model
agent.load_model('Models/dqn_model.pth')  # Update the path as necessary

# Run the test
test_trained_model(env, agent, episodes=1)  # Adjust the number of test episodes as needed
test_trained_model_with_visualization(env, agent, episodes=1)