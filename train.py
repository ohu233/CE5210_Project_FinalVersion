from Crawler import BusCrawler
from Env import BusEnv
from DQN import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import math
from DQN import DuelingDoubleDQNAgent

def train_dqn(env, agent, time, episodes):
    rewards = []  # 用于存储每个 episode 的总奖励值
    plt.ion()  # 开启 matplotlib 的交互模式以实现动态更新

    for e in tqdm(range(episodes), desc="Training Episodes", leave=False):
        state = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state)

            to_old_memory = (e % 40 == 0)
            agent.remember(state, action, reward, next_state, done, to_old_memory=to_old_memory)
            state = next_state
            total_reward += reward
            agent.replay()  # 训练模型
            
        tqdm.write(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        rewards.append(total_reward)  # 记录当前 episode 的总奖励

        # 进行 epsilon 衰减
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # 更新 target 网络
        if e % 8 == 0:
            agent.update_target_network()

        # 绘制实时奖励变化曲线
        plot_rewards(rewards)
    
    print("Training completed.")
    save_results(rewards, time)

    # 保存模型和经验回放缓冲区
    model_path = f"Models/dqn_model.pth"
    os.makedirs('Models', exist_ok=True)
    agent.save_model(model_path)
    print(f"模型和经验回放数据已保存到: {model_path}")

def plot_rewards(rewards, window_size=100):
    plt.clf()  # 清除当前图形
    x_values = range(len(rewards))

    # 对于前100个 episode，只绘制每个 episode 的奖励值，不计算均值和标准差
    if len(rewards) <= window_size:
        plt.plot(x_values, rewards, linestyle='-', color='b', alpha=0.1, label='Reward')
    else:
        # 对于超过100个 episode 的情况，继续原始绘制方式
        plt.plot(x_values, rewards, linestyle='-', color='b', alpha=0.1, label='Reward')

        # 计算滑动平均值和标准差
        smoothed_rewards = []
        std_devs = []  # 存储标准差

        for i in range(len(rewards)):
            if i < window_size:
                # 对于不足滑动窗口大小的奖励数，计算当前所有奖励的平均值和标准差
                window_data = rewards[:i + 1]
            else:
                # 奖励数达到窗口大小后，使用滑动窗口的平均值和标准差
                window_data = rewards[i - window_size + 1:i + 1]

            avg_reward = np.mean(window_data)
            std_dev = np.std(window_data)

            smoothed_rewards.append(avg_reward)
            std_devs.append(std_dev)

        # 绘制从第101个 episode 开始的平均奖励变化曲线
        plt.plot(x_values[window_size:], smoothed_rewards[window_size:], linestyle='-', color='r', label='Moving Average Reward')

        # 绘制波动范围（均值 ± 标准差）
        lower_bound = np.array(smoothed_rewards) - np.array(std_devs)
        upper_bound = np.array(smoothed_rewards) + np.array(std_devs)
        plt.fill_between(x_values[window_size:], lower_bound[window_size:], upper_bound[window_size:], color='blue', alpha=0.3, label='Reward ± Std Dev')

    # 设置图形样式
    plt.title("Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.pause(0.001)  # 短暂停以刷新图表

# 保存训练结果和奖励图
def save_results(rewards, time):
    # 创建结果文件夹
    os.makedirs('Figures', exist_ok=True)
    
    # 保存奖励数据
    rewards_file = f'Figures/data/rewards_{time}.csv'
    np.savetxt(rewards_file, rewards, delimiter=",")
    
    # 绘制并保存最终的奖励曲线
    plt.ioff()  # 关闭交互模式
    plt.figure()
    plot_rewards(rewards)  # 调用绘图函数以确保图像为最终结果
    plt.savefig(f'Figures/fig/reward_{time}.png')

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
data_file = f'Crawler/data_{formatted_datetime}.csv'
data.to_csv(data_file, index=False)
print(f"数据已保存到: {data_file}")
'''

all_data = pd.read_excel('Data/preprocessed.xlsx')
uni = all_data['CollectTime'].unique()
data = all_data[all_data['CollectTime'] == uni[1]]
'''

# 创建环境和 DQN Agent
env = BusEnv(data)
agent = DuelingDoubleDQNAgent(state_size=env.state_dim, action_size=env.action_space)

# 开始训练
train_dqn(env, agent, formatted_datetime, episodes=2000)