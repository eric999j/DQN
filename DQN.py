import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

'''
一個深度 Q 網絡 (DQN) 代理的 Python 實現，用於解決 CartPole 環境中的控制問題。
'''

# 用於表示儲存在經驗回放緩衝區中的轉換 (state, action, next_state, reward)。
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """實現一個經驗回放緩衝區，用於儲存和取樣轉換，以訓練 DQN 代理。"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward):
        """存储经验"""
        self.buffer.append(Transition(state, action, next_state, reward))
        
    def sample(self, batch_size):
        """随机采样"""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """
    定義深度 Q 網絡模型，該模型是一個具有三個全連接層的神經網絡。
    它接收一個狀態作為輸入，並輸出每個可能動作的 Q 值。
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 定义网络层
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """定义前向传播"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    """實現 DQN 代理，包括選擇動作、學習和更新網絡的方法。"""
    def __init__(self, state_dim, action_dim, replay_capacity=10000, batch_size=128, gamma=0.99, lr=1e-4, tau=0.005, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma # 折扣因子
        self.lr = lr # 学习率
        self.tau = tau # 目标网络软更新参数
        self.epsilon = epsilon_start # Epsilon-greedy 策略的起始探索率
        self.epsilon_end = epsilon_end # Epsilon-greedy 策略的最终探索率
        self.epsilon_decay = epsilon_decay # Epsilon-greedy 策略的衰减率

        # 初始化策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 复制策略网络的权重到目标网络
        self.target_net.eval() # 将目标网络设置为评估模式

        # 初始化优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # 初始化经验回放缓冲区
        self.memory = ReplayMemory(replay_capacity)

        self.steps_done = 0 # 记录已经执行的步数，用于 epsilon 衰减

    def select_action(self, state):
        """epsilon-greedy 策略选择动作"""
        self.steps_done += 1
        if random.random() > self._current_epsilon():
            return self._greedy_action(state)
        return self._random_action()

    def _current_epsilon(self):
        """计算指数衰减的epsilon值"""
        return self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

    def _greedy_action(self, state):
        """策略网络选择最优动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state_tensor).argmax().view(1, 1)

    def _random_action(self):
        """随机探索动作"""
        return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

    def learn(self):
        """从经验回放缓冲区中采样并更新网络"""
        if len(self.memory) < self.batch_size:
            return # 如果缓冲区中的样本数量不足，则不进行学习

        # 从缓冲区中采样
        transitions = self.memory.sample(self.batch_size)
        # 将一批转换（Transition 对象）转换为一个 Transition 对象，其中每个字段都是一个包含所有样本的张量
        # 例如，batch.state 将是一个包含所有状态张量的张量
        batch = Transition(*zip(*transitions))

        # 计算非最终状态的掩码，并将它们连接成一个张量
        # next_state 为 None 表示这是一个最终状态
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        # 将所有非最终状态的 next_state 连接成一个张量
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                                           for s in batch.next_state if s is not None])

        # 将状态、动作和奖励连接成张量
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat([torch.tensor([r], dtype=torch.float32) for r in batch.reward])

        # 计算当前状态下所选动作的 Q 值 (Q(s_t, a_t))
        # policy_net 计算 Q(s_t, :)，然后我们选择 action_batch 中指定的动作的 Q 值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算下一个状态的 Q 值的期望 (V(s_{t+1}))
        # 对于所有非最终的下一个状态，计算 V(s_{t+1}) = max_a Q_{target}(s_{t+1}, a)
        # 对于最终状态，其 V(s_{t+1}) = 0
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad(): # 不计算梯度
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # 计算期望的 Q 值 (y_t = r_t + gamma * V(s_{t+1}))
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze()

        # 计算损失 (Huber loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad() # 清除之前的梯度
        loss.backward() # 计算梯度
        # 对策略网络的梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step() # 更新网络权重

        # 软更新目标网络的权重
        # θ_target = τ * θ_policy + (1 - τ) * θ_target
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

import gymnasium as gym
import math

# --- 训练循环 ---
if __name__ == '__main__':
    # 创建 CartPole 环境
    # render_mode="human" 可以可视化训练过程，但会降低速度
    env = gym.make("CartPole-v1") #, render_mode="human")

    # 获取状态和动作空间的维度
    state, info = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    # 初始化 DQN 代理
    agent = DQNAgent(state_dim, action_dim)

    # 训练参数
    num_episodes = 600
    episode_durations = [] # 记录每个 episode 的持续时间（步数）

    print(f"Starting training for {num_episodes} episodes...")

    for i_episode in range(num_episodes):
        # 初始化环境和状态
        state, info = env.reset()
        state = np.array(state, dtype=np.float32) # 保持为numpy数组但移除多余的维度

        total_reward = 0
        terminated = False
        truncated = False
        t = 0
        while not terminated and not truncated:
            # 选择并执行动作
            action_tensor = agent.select_action(state)
            action = action_tensor.item()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            # 处理下一个状态
            if terminated:
                next_state = None
            else:
                next_state = np.array(observation, dtype=np.float32) # 保持维度一致

            # 将转换存储在经验回放缓冲区中
            # 注意：我们将 state 和 next_state 存储为 numpy 数组，因为 select_action 需要 numpy
            # 但在 learn 方法内部，它们会被转换回 Tensor
            agent.memory.store(state, action_tensor, next_state, reward_tensor)

            # 移动到下一个状态
            state = next_state

            # 执行一步优化 (在策略网络上)
            agent.learn()

            t += 1
            if terminated or truncated:
                episode_durations.append(t + 1)
                if (i_episode + 1) % 50 == 0: # 每 50 个 episode 打印一次信息
                     print(f"Episode {i_episode+1}/{num_episodes} finished after {t+1} timesteps. Total reward: {total_reward:.2f}. Epsilon: {agent.epsilon:.3f}")
                break

    print('Training complete.')
    env.close()

    # 可以添加绘图代码来可视化 episode_durations
    import matplotlib.pyplot as plt
    plt.plot(episode_durations)
    plt.title('Episode Durations over Time')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.show()
