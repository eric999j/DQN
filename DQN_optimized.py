import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import time
import gymnasium as gym
import matplotlib.pyplot as plt

'''
一個深度 Q 網絡 (DQN) 代理的 Python 實現，用於解決 CartPole 環境中的控制問題。
優化版本，包含多項性能改進。
'''

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 用於表示儲存在經驗回放緩衝區中的轉換 (state, action, next_state, reward, done)。
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """實現一個經驗回放緩衝區，用於儲存和取樣轉換，以訓練 DQN 代理。"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        """存储经验"""
        self.buffer.append(Transition(state, action, next_state, reward, done))
        
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
        # 定义网络层 - 使用更高效的序列模型
        self.network = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        """定义前向传播"""
        return self.network(x)

class DQNAgent:
    """實現 DQN 代理，包括選擇動作、學習和更新網絡的方法。"""
    def __init__(self, state_dim, action_dim, replay_capacity=10000, batch_size=128, 
                 gamma=0.99, lr=1e-3, tau=0.005, epsilon_start=0.9, 
                 epsilon_end=0.05, epsilon_decay=1000, update_every=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma # 折扣因子
        self.lr = lr # 学习率
        self.tau = tau # 目标网络软更新参数
        self.epsilon = epsilon_start # Epsilon-greedy 策略的起始探索率
        self.epsilon_end = epsilon_end # Epsilon-greedy 策略的最终探索率
        self.epsilon_decay = epsilon_decay # Epsilon-greedy 策略的衰减率
        self.update_every = update_every # 每隔多少步更新一次网络

        # 初始化策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 复制策略网络的权重到目标网络
        self.target_net.eval() # 将目标网络设置为评估模式

        # 初始化优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # 初始化经验回放缓冲区
        self.memory = ReplayMemory(replay_capacity)

        self.steps_done = 0 # 记录已经执行的步数，用于 epsilon 衰减
        self.learn_step_counter = 0 # 用于控制目标网络更新频率

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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_tensor).argmax().view(1, 1)

    def _random_action(self):
        """随机探索动作"""
        return torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)

    def learn(self):
        """从经验回放缓冲区中采样并更新网络"""
        if len(self.memory) < self.batch_size:
            return # 如果缓冲区中的样本数量不足，则不进行学习
            
        self.learn_step_counter += 1
        
        # 只在特定步数更新网络，减少计算量
        if self.learn_step_counter % self.update_every != 0:
            return

        # 从缓冲区中采样
        transitions = self.memory.sample(self.batch_size)
        # 将一批转换（Transition 对象）转换为一个 Transition 对象，其中每个字段都是一个包含所有样本的张量
        batch = Transition(*zip(*transitions))

        # 使用 done 标志来处理终止状态
        done_mask = torch.tensor(batch.done, dtype=torch.bool)
        
        # 批量处理状态和动作，减少循环和张量创建操作
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        
        # 处理非终止状态的下一个状态
        non_final_mask = ~done_mask
        non_final_next_states = torch.tensor(
            np.array([s for s, d in zip(batch.next_state, batch.done) if not d]), 
            dtype=torch.float32
        ).to(device)

        # 计算当前状态下所选动作的 Q 值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算下一个状态的 Q 值的期望
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states.size(0) > 0:  # 确保有非终止状态
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
                
        # 计算期望的 Q 值
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 计算损失 (Huber loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # 软更新目标网络的权重
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss.item()

# --- 训练循环 ---
if __name__ == '__main__':
    start_time = time.time()
    
    # 创建 CartPole 环境
    env = gym.make("CartPole-v1")

    # 获取状态和动作空间的维度
    state, info = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    # 初始化 DQN 代理
    agent = DQNAgent(state_dim, action_dim, batch_size=64, update_every=4)

    # 训练参数
    num_episodes = 600
    episode_durations = []
    rewards = []
    losses = []

    print(f"Starting training for {num_episodes} episodes...")

    for i_episode in range(num_episodes):
        episode_start = time.time()
        
        # 初始化环境和状态
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)

        total_reward = 0
        episode_loss = []
        terminated = False
        truncated = False
        t = 0
        
        while not terminated and not truncated:
            # 选择并执行动作
            action_tensor = agent.select_action(state)
            action = action_tensor.item()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # 处理下一个状态
            next_state = np.array(observation, dtype=np.float32) if not terminated else None
            done = terminated or truncated

            # 将转换存储在经验回放缓冲区中
            agent.memory.store(state, action_tensor, next_state, reward, done)

            # 移动到下一个状态
            state = next_state

            # 执行一步优化
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            t += 1
            if done:
                episode_durations.append(t + 1)
                rewards.append(total_reward)
                if episode_loss:
                    losses.append(sum(episode_loss) / len(episode_loss))
                
                if (i_episode + 1) % 50 == 0:
                    episode_time = time.time() - episode_start
                    print(f"Episode {i_episode+1}/{num_episodes} finished after {t+1} timesteps. "
                          f"Total reward: {total_reward:.2f}. Epsilon: {agent._current_epsilon():.3f}. "
                          f"Time: {episode_time:.2f}s")
                break

    total_time = time.time() - start_time
    print(f'Training complete. Total time: {total_time:.2f} seconds')
    env.close()

    # 绘图代码来可视化训练结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_durations)
    plt.title('Episode Durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    
    plt.subplot(1, 3, 2)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    if losses:
        plt.subplot(1, 3, 3)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()