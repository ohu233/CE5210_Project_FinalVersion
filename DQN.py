import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import time
from Env import BusEnv
import pickle

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义 Dueling DQN 网络
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc_value = nn.Linear(256, 256)
        self.fc_advantage = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)  # 状态价值分支
        self.advantage = nn.Linear(256, action_size)  # 优势分支

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        # 计算状态价值和优势
        value = torch.relu(self.fc_value(x))
        value = self.value(value)
        
        advantage = torch.relu(self.fc_advantage(x))
        advantage = self.advantage(advantage)
        
        # 合并状态价值和优势
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    

class LayeredMemory:
    def __init__(self, new_memory_size, old_memory_size, memory_file):
        self.new_memory = deque(maxlen=new_memory_size)  # 新数据区，用于存储最新的经验
        self.old_memory = deque(maxlen=old_memory_size)  # 旧数据区，用于保留旧的经验
        self.memory_file = memory_file  # 存储经验池的文件路径

        # 尝试从文件加载经验池数据
        self.load_memory()

    def add(self, experience, to_old_memory=False):
        """将新的经验加入经验池"""
        if to_old_memory:
            self.old_memory.append(experience)  # 加入旧数据区
        else:
            self.new_memory.append(experience)  # 加入新数据区

    def sample(self, batch_size):
        """从新数据区和旧数据区采样一个批次"""
        new_batch_size = int(batch_size * 0.7)  # 按比例采样
        old_batch_size = batch_size - new_batch_size

        # 如果新数据区不够，则用旧数据区补充，反之亦然
        if len(self.new_memory) < new_batch_size:
            new_samples = list(self.new_memory)  # 获取全部新数据区内容
            old_samples = random.sample(self.old_memory, batch_size - len(new_samples))
        elif len(self.old_memory) < old_batch_size:
            old_samples = list(self.old_memory)
            new_samples = random.sample(self.new_memory, batch_size - len(old_samples))
        else:
            new_samples = random.sample(self.new_memory, new_batch_size)
            old_samples = random.sample(self.old_memory, old_batch_size)

        return new_samples + old_samples

    def save_memory(self):
        """将经验池保存到文件"""
        with open(self.memory_file, 'wb') as f:
            pickle.dump((self.new_memory, self.old_memory), f)
        print("经验池已保存到文件。")

    def load_memory(self):
        """从文件加载经验池"""
        try:
            with open(self.memory_file, 'rb') as f:
                self.new_memory, self.old_memory = pickle.load(f)
            print("经验池已从文件加载。")
        except (FileNotFoundError, EOFError):
            print("未找到经验池文件或文件为空，初始化空的经验池。")

    def __len__(self):
        """返回新数据区和旧数据区的总长度"""
        return len(self.new_memory) + len(self.old_memory)
    
class DuelingDoubleDQNAgent:
    def __init__(self, state_size, action_size, memory_file='Memory/memory.pkl'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = LayeredMemory(new_memory_size=6000, old_memory_size=4000, memory_file=memory_file)
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.003
        self.batch_size = 128
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.device = device

        # 定义 Dueling Q 网络结构
        self.q_network = DuelingQNetwork(state_size, action_size).to(device)
        self.target_network = DuelingQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # 初始化目标网络
        self.update_target_network()

    def update_target_network(self):
        """更新目标网络的权重"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done, to_old_memory=False):
        """将经验加入分层经验池"""
        self.memory.add((state, action, reward, next_state, done), to_old_memory=to_old_memory)

    def act(self, state):
        """使用 ε-贪心策略选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.q_network(state)
        return torch.argmax(act_values).item()

    def replay(self):
        """从经验池中采样数据并训练模型"""
        if len(self.memory) < self.batch_size:
            return

        # 从分层经验池中采样
        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 当前 Q 值
        current_qs = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN 计算目标 Q 值
        next_actions = torch.argmax(self.q_network(next_states), dim=1)
        next_qs = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_qs = rewards + (1 - dones) * self.gamma * next_qs

        loss = nn.functional.mse_loss(current_qs, target_qs.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        '''       
         # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        '''

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)
        # 训练完成后保存经验池
        self.memory.save_memory()

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
        self.q_network.eval()

# 定义 DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.003
        self.batch_size = 128
        device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )
        self.device = device

        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.q_network(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 转换为 numpy 数组后再转换为 GPU 张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 当前 Q 值
        current_qs = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 下一个状态的最大 Q 值
        max_next_qs = self.target_network(next_states).max(1)[0]
        target_qs = rewards + (1 - dones) * self.gamma * max_next_qs

        # 计算损失并优化
        loss = nn.functional.mse_loss(current_qs, target_qs.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)  # 保存模型权重到指定路径

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))  # 加载模型权重
        self.q_network.eval()  # 设置模型为评估模式
