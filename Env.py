import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

class BusEnv:
    '''
    定义 BusEnv 环境
    称车站为泊位，泊位容量即能够同时服务的车辆数
    '''
    def __init__(self, data):
        self.capacity = 2   # 泊位容量
        self.bay_num = 2    # 泊位数量
        self.service_time = 50  #标准车辆服务时间，在step（）以及reward（）中会分别对每辆车进行修正
        self.data = data    # 数据加载，由crawler爬取

        self.bays = [[None] * self.capacity for _ in range(self.bay_num)]   # 初始化泊位为[[None,None],[None,None]]
        self.waiting_queue = [[] for _ in range(self.bay_num)]  # 初始化等待队列为[[],[]]
        self.current_step = 0   # 初始化当前步为0
        self.last_event_time = None # 初始化时间点记录

        # 动作空间定义：泊位分配，仅包含 0 和 1，分别表示 bay1 和 bay2（需要修改丰富）
        self.action_space = 2

        # 4 个泊位状态 + 2个泊位等待队列长度 + 6 个车辆信息 + bool(是否高峰时间) + 4个泊位时间差
        self.state_dim = 16

        # 初始化线路分配，按照City Hall Exit B实际情况的公交线路分配
        self.initial_assignment = {
                                    '1':'0',
                                    '2':'0',
                                    '3':'0',
                                    '4':'0',
                                    '5':'1',
                                    '6':'0',
                                    '7':'1',
                                    '8':'1',
                                    '9':'0',
                                    '10':'1',
                                    '11':'1',
                                    '12':'1',
                                    '13':'1'
        }
    
    
    def reset(self):
        # 重置泊位状态、等待队列、步数及时间记录变量
        self.bays = [[None] * self.capacity for _ in range(self.bay_num)]
        self.waiting_queue = [[] for _ in range(self.bay_num)]
        self.current_step = 0
        self.last_event_time = None

        # 返回初始状态观察值
        return self._next_observation()

    def _next_observation(self):
        # 获取泊位状态信息
        bay_status = np.array(
            [1 if self.bays[i][j] is not None else 0 for i in range(self.bay_num) for j in range(self.capacity)],
            dtype=np.float32
        )

        # 获取每个泊位的等待队列长度
        waiting_queue_length = np.array(
            [len(queue) for queue in self.waiting_queue], dtype=np.float32
        )

        # 获取当前车辆信息       
        # 服务时间调整参数
        morning_peak = (datetime.strptime('07:00', '%H:%M').time(), datetime.strptime('09:00', '%H:%M').time())
        evening_peak = (datetime.strptime('17:00', '%H:%M').time(), datetime.strptime('19:00', '%H:%M').time())

        if self.current_step < len(self.data):  # 检查是否数据处理完毕
            next_vehicle = self.data.iloc[self.current_step]

            estimatedarrival = pd.to_datetime(next_vehicle['EstimatedArrival']).time()     
            peak_flow = 1 if (morning_peak[0] <= estimatedarrival <= morning_peak[1]) or (evening_peak[0] <= estimatedarrival <= evening_peak[1]) else 0

            vehicle_info = np.array([
                next_vehicle['ServiceNo'], 
                next_vehicle['Load'], 
                next_vehicle['Type'],
                next_vehicle['Latitude'], 
                next_vehicle['Longitude'],
                peak_flow], 
                dtype=np.float32)
            
            estimated_arrival = pd.to_datetime(next_vehicle['EstimatedArrival'])
        else:   # 否则设定为0，即无车辆到达
            vehicle_info = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
            estimated_arrival = None  # 表示没有更多车辆

        # 计算泊位中车辆的离开时间差值（如果有当前处理车辆）
        departure_time_diffs = []
        for bay in self.bays:
            for slot in bay:
                if slot is not None and estimated_arrival is not None:
                    # 计算离开时间与当前车辆到达时间的差值（单位为秒）
                    time_diff = (slot['DepartureTime'] - estimated_arrival).total_seconds()
                else:
                    # 如果泊位为空或没有当前处理车辆，时间差设为0
                    time_diff = 0
                departure_time_diffs.append(time_diff)
        
        # 将离开时间差值转换为 numpy 数组
        departure_time_diffs = np.array(departure_time_diffs, dtype=np.float32)

        # 合并泊位状态、等待队列长度、离开时间差和车辆信息
        return np.concatenate([bay_status, waiting_queue_length, departure_time_diffs, vehicle_info])
    
    def step(self, action):
        # 如果数据处理完毕
        if self.current_step >= len(self.data):
            return self._next_observation(), 0, True  # 没有更多车辆，结束 episode

        # 获取当前车辆信息
        row = self.data.iloc[self.current_step]
        vehicle_info = {
            'ServiceNo': row['ServiceNo'],
            'Load': row['Load'],
            'Type': row['Type'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'VehCode': row['VehCode'],
            'EstimatedArrival': row['EstimatedArrival'],
            'AllocatedTime': None,
            'DepartureTime': None
        }
        # 初始化当前step总奖励
        total_reward = 0

        # 检查并释放所有泊位中已完成服务的车辆
        for bay_index in range(self.bay_num):
            for slot_index in range(self.capacity):
                if (self.bays[bay_index][slot_index] and 
                    self.bays[bay_index][slot_index]['DepartureTime'] <= vehicle_info['EstimatedArrival']):
                    self.last_event_time = self.bays[bay_index][slot_index]['DepartureTime']
                    self.bays[bay_index][slot_index] = None  
                    if self.waiting_queue[bay_index]:
                        next_bus = self.waiting_queue[bay_index].pop(0)
                        next_bus['AllocatedTime'] = self.last_event_time
                        next_bus['DepartureTime'] = next_bus['AllocatedTime'] + pd.Timedelta(seconds=self.service_time)
                        self.bays[bay_index][slot_index] = next_bus
                        '''
                        bay排队超过一定长度后进行惩罚
                        '''
                        reward = -0.5 * ((next_bus['AllocatedTime'] - next_bus['EstimatedArrival']).total_seconds())**2
                        if reward == 0:
                            reward = 50
                        total_reward += reward

        # 检查选定泊位是否有空位
        allocated = False
        if None in self.bays[action]:
            empty_slot = self.bays[action].index(None)
            if self.waiting_queue[action]:
                next_bus = self.waiting_queue[action].pop(0)
                next_bus['AllocatedTime'] = max(next_bus['EstimatedArrival'], vehicle_info['EstimatedArrival'])
            else:
                next_bus = vehicle_info
                next_bus['AllocatedTime'] = next_bus['EstimatedArrival']

            '''
            离开 <- area1 area2 <- area1 area2 <- 到达
                      0    1    
            该情况应当给予惩罚：由于后到达的车辆不好进入靠前的泊位

            检查是否为[None, 1]的情况并给予惩罚
            '''
            
            bay_state = [1 if self.bays[action][i] is not None else 0 for i in range(self.capacity)]
            if bay_state == [0, 1] and empty_slot == 0:
                total_reward -= 50  # 惩罚值可根据需要调整

            # 获取车辆的当前初始泊位
            preferred_bay = self.initial_assignment.get(vehicle_info['ServiceNo'], None)

            '''
            同一路线尽量减少变动
            路径一致性奖励/惩罚和状态更新
            '''
            if preferred_bay is not None:
                if preferred_bay == action:
                    # 如果泊位一致，奖励
                    total_reward += 30  # 路径一致性奖励
                else:
                    # 如果泊位不一致，惩罚并更新初始分配
                    total_reward -= 50  # 路径不一致惩罚
                    self.initial_assignment[vehicle_info['ServiceNo']] = action  # 更新初始分配为当前泊位
            else:
                # 没有指定初始泊位的线路，初始化并设置当前泊位为初始泊位
                self.initial_assignment[vehicle_info['ServiceNo']] = action

            # 设置车辆离场时间
            next_bus['DepartureTime'] = next_bus['AllocatedTime'] + pd.Timedelta(seconds=self.service_time)
            self.bays[action][empty_slot] = next_bus
            self.last_event_time = next_bus['AllocatedTime']
            reward = -0.5 * ((next_bus['AllocatedTime'] - next_bus['EstimatedArrival']).total_seconds())**2
            if reward == 0:
                reward = 50
            total_reward += reward
            allocated = True

        if not allocated:
            # 所有泊位已满，加入等待队列
            self.waiting_queue[action].append(vehicle_info)
            total_reward -= np.exp(len(self.waiting_queue[action]))  # 惩罚车辆进入等待队列
            self.last_event_time = vehicle_info['EstimatedArrival']

        # 更新到下一辆车
        self.current_step += 1
        done = self.current_step >= len(self.data)

        return self._next_observation(), total_reward, done
    
    def para(self, veh):

        load, type = veh['Load'], veh['Type']


        alpha = 0.9 if type == '1' else 1.1
        beta = 0.9 if load == '1' else 1 if load == '2' else 1.1
        '''        
        gamma = 0.9 if dayofweek in ['6', '7'] else 1
        '''

        return alpha, beta#, gamma, theta

    def reward(self):
        '''
        info作为系数去影响奖励
        '''
        reward = 0
        for i in range(self.bay_num):
            for j in range(self.capacity):
                if self.bays[i][j] is not None:
                    alpha, beta = self.para(self.bays[i][j])
                    reward += beta * 20
        for i in range(self.bay_num):
            for bus in self.waiting_queue[i]:
                alpha, beta = self.para(bus)
                reward -= 10 * alpha * beta

        return float(reward)
    
    def random_action_test(self, episodes=10):
        """随机动作验证，输出分配结果"""
        for episode in range(episodes):
            state = self.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                action = random.choice([0, 1])  # 随机选择一个泊位
                state, reward, done, _ = self.step(action)
                total_reward += reward
                step += 1

                # 打印每一步的状态信息，包括泊位和等待队列情况
                print(f"Episode: {episode + 1}, Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")
                print("Current bay status:")
                for i in range(self.bay_num):
                    bay_status = [veh['ServiceNo'] if veh else None for veh in self.bays[i]]
                    print(f"Bay {i+1}: {bay_status}")
                
                print("Current waiting queue:")
                for i in range(self.bay_num):
                    queue_status = [veh['ServiceNo'] for veh in self.waiting_queue[i]]
                    print(f"Queue {i+1}: {queue_status}")
                
                print('-'*100)

            print(f"Episode {episode + 1} finished with total reward: {total_reward}\n")


'''
all_data = pd.read_excel('Data/preprocessed.xlsx')
uni = all_data['CollectTime'].unique()
data = all_data[all_data['CollectTime'] == uni[0]]

# 创建环境并运行随机动作测试
env = BusEnv(data)
env.random_action_test(episodes=1)
'''
