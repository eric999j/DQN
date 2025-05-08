import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import time
import gymnasium as gym # gymnasium is the updated gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any

'''
一個深度 Q 網絡 (DQN) 代理的 Python 實現，用於解決 CartPole 環境中的控制問題。
優化版本，包含多項性能改進。

* 型別提示 (Type Hinting): 在整個程式碼中加入了型別提示
    * 例如 capacity: int, state: np.ndarray, -> Optional[float]。這有助於靜態分析工具 (如 MyPy) 檢查型別錯誤，並提高程式碼的可讀性與可維護性。

* ReplayMemory.store 中 action 的型別:
    * 原先 store 方法可能接收一個 torch.Tensor 作為 action。現修改為接收一個 int 型別的 action。這使得經驗回放緩衝區儲存的資料型別更為基礎和通用。
    * 相應地，在訓練迴圈中，agent.memory.store(state, action_tensor.item(), ...) 將 action_tensor 轉換為 int。

* DQNAgent.learn 中批次資料的處理:
    * action_batch: 由於 batch.action 現在是一個 int 的元組 (tuple of ints)，創建 action_batch 的方式調整為 torch.tensor(batch.action, device=device, dtype=torch.long).unsqueeze(1)。
    * state_batch, reward_batch, non_final_next_states_tensor: 使用 torch.from_numpy(np.array(...)) 或 torch.tensor(...) 創建張量。np.array(batch.state) 會將 NumPy 陣列的元組合併成一個更高維度的 NumPy 陣列，然後轉換為 PyTorch 張量。
    * next_state_values 的計算: 原邏輯已相當清晰。微調了 non_final_next_states_list 的創建，以明確過濾掉 None 值，然後再轉換為張量。max(1).values 用於獲取最大 Q 值，取代原來的 max(1)[0]，更具可讀性。
    * expected_state_action_values 的計算: 確保 reward_batch 和 next_state_values.unsqueeze(1) 的形狀一致 ([batch_size, 1]) 以進行正確的元素級相加。

* 訓練迴圈中的狀態處理:
    * env.reset() 在 gymnasium 中返回一個元組 (observation, info)。已更新為 state_tuple = env.reset(); state = state_tuple[0]。
    * 移除了對 state 和 observation 的 np.array(..., dtype=np.float32) 轉換，因為 CartPole 環境本身返回的觀測值就是 np.ndarray 且型別為 np.float32。
    * next_state 在 terminated 時設為 None，這對於經驗回放是正確的。
    * 在 agent.memory.store 時，action 參數傳遞的是 action_tensor.item()。
    * 日誌中的 epsilon 現在直接使用 agent.epsilon，它會在 select_action 中被更新。

* DQNAgent._current_epsilon:
    * 在 select_action 中更新 self.epsilon = self._current_epsilon()，使得 self.epsilon 屬性始終反映當前的探索率。
    * _current_epsilon 的計算現在明確使用 self.epsilon_start 作為衰減的初始值，避免了 self.epsilon 被重複衰減的問題。

* 程式碼小細節:
    * ReplayMemory 初始化 deque 時使用 deque([], maxlen=capacity) 以提供一個空的初始可迭代物件。
    * 更新了日誌輸出格式，使其更易讀。
    * 繪圖部分使用了 fig, axs = plt.subplots(...)，對子圖有更好的控制。
    * 變數名 losses 改為 avg_losses 以更準確反映其內容 (每回合的平均損失) 

'''

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 用於表示儲存在經驗回放緩衝區中的轉換 (state, action, next_state, reward, done)。
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """實現一個經驗回放緩衝區，用於儲存和取樣轉換，以訓練 DQN 代理。"""
    def __init__(self, capacity: int):
        self.buffer = deque([], maxlen=capacity) # Initialize with an empty iterable
        
    def store(self, state: np.ndarray, action: int, next_state: Optional[np.ndarray], reward: float, done: bool) -> None:
        """存储经验"""
        self.buffer.append(Transition(state, action, next_state, reward, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """随机采样"""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    """
    定義深度 Q 網絡模型，該模型是一個具有三個全連接層的神經網絡。
    它接收一個狀態作為輸入，並輸出每個可能動作的 Q 值。
    """
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        # 定义网络层 - 使用更高效的序列模型
        self.network = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义前向传播"""
        return self.network(x)

class DQNAgent:
    """實現 DQN 代理，包括選擇動作、學習和更新網絡的方法。"""
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 replay_capacity: int = 10000, 
                 batch_size: int = 128, 
                 gamma: float = 0.99, 
                 lr: float = 1e-3, 
                 tau: float = 0.005, 
                 epsilon_start: float = 0.9, 
                 epsilon_end: float = 0.05, 
                 epsilon_decay: float = 1000, 
                 update_every: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma # 折扣因子
        self.lr = lr # 学习率
        self.tau = tau # 目标网络软更新参数
        self.epsilon_start = epsilon_start # Epsilon-greedy 策略的起始探索率 (保存原始起始值)
        self.epsilon = epsilon_start # 当前的 Epsilon
        self.epsilon_end = epsilon_end # Epsilon-greedy 策略的最终探索率
        self.epsilon_decay = epsilon_decay # Epsilon-greedy 策略的衰减率
        self.update_every = update_every # 每隔多少步更新一次网络

        # 初始化策略网络和目标网络
        self.policy_net: DQN = DQN(state_dim, action_dim).to(device)
        self.target_net: DQN = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 复制策略网络的权重到目标网络
        self.target_net.eval() # 将目标网络设置为评估模式

        # 初始化优化器
        self.optimizer: optim.AdamW = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # 初始化经验回放缓冲区
        self.memory: ReplayMemory = ReplayMemory(replay_capacity)

        self.steps_done: int = 0 # 记录已经执行的步数，用于 epsilon 衰减
        self.learn_step_counter: int = 0 # 用于控制目标网络更新频率

    def select_action(self, state: np.ndarray) -> torch.Tensor: # Returns tensor for consistency, will be .item()ed for env
        """epsilon-greedy 策略选择动作"""
        self.steps_done += 1
        self.epsilon = self._current_epsilon() # Update current epsilon
        if random.random() > self.epsilon:
            return self._greedy_action(state)
        return self._random_action()

    def _current_epsilon(self) -> float:
        """计算指数衰减的epsilon值"""
        # self.epsilon_start is used here, not self.epsilon, to ensure decay is from original start
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

    def _greedy_action(self, state: np.ndarray) -> torch.Tensor:
        """策略网络选择最优动作"""
        with torch.no_grad():
            # Convert state to tensor, add batch dimension, move to device
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # Get Q-values, then select action with max Q-value
            return self.policy_net(state_tensor).argmax().view(1, 1)

    def _random_action(self) -> torch.Tensor:
        """随机探索动作"""
        return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)

    def learn(self) -> Optional[float]:
        """从经验回放缓冲区中采样并更新网络"""
        if len(self.memory) < self.batch_size:
            return None # 如果缓冲区中的样本数量不足，则不进行学习
            
        self.learn_step_counter += 1
        
        # 只在特定步数更新网络，减少计算量
        if self.learn_step_counter % self.update_every != 0:
            return None

        # 从缓冲区中采样
        transitions: List[Transition] = self.memory.sample(self.batch_size)
        # 将一批转换（Transition 对象）转换为一个 Transition 对象，其中每个字段都是一个包含所有样本的元组
        batch: Transition = Transition(*zip(*transitions))

        # batch.state is a tuple of np.ndarrays. Convert to a single 2D np.ndarray then to tensor.
        state_batch: torch.Tensor = torch.from_numpy(np.array(batch.state)).float().to(device)
        
        # batch.action is a tuple of ints. Convert to a tensor.
        action_batch: torch.Tensor = torch.tensor(batch.action, device=device, dtype=torch.long).unsqueeze(1)
        
        # batch.reward is a tuple of floats. Convert to a tensor.
        reward_batch: torch.Tensor = torch.tensor(batch.reward, device=device, dtype=torch.float32).unsqueeze(1)
        
        # batch.done is a tuple of bools. Convert to a tensor.
        done_batch: torch.Tensor = torch.tensor(batch.done, device=device, dtype=torch.bool) # shape: [batch_size]

        # 处理非终止状态的下一个状态
        # Create a mask for non-final states
        non_final_mask = ~done_batch # Invert done_batch

        # Collect non-final next states. batch.next_state contains None for terminal states.
        # np.array will handle a list of arrays or Nones, but we filter Nones first.
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        
        # Compute Q values for next states
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states_list: # Check if the list is not empty
            non_final_next_states_tensor = torch.from_numpy(np.array(non_final_next_states_list)).float().to(device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states_tensor).max(1).values
        
        # 计算期望的 Q 值 (Bellman equation)
        # Unsqueeze next_state_values to make it [batch_size, 1] for broadcasting with reward_batch
        expected_state_action_values = reward_batch + (self.gamma * next_state_values.unsqueeze(1))
        
        # 计算当前状态下所选动作的 Q 值
        # policy_net(state_batch) output is [batch_size, n_actions]
        # action_batch is [batch_size, 1] (indices of chosen actions)
        # .gather(1, action_batch) selects Q-values for chosen actions, result is [batch_size, 1]
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 计算损失 (Huber loss)
        criterion = nn.SmoothL1Loss()
        loss: torch.Tensor = criterion(state_action_values, expected_state_action_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # Clip gradients
        self.optimizer.step()

        # 软更新目标网络的权重
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
        return loss.item()

# --- 訓練循環 ---
if __name__ == '__main__':
    start_time = time.time()
    
    # 創建 CartPole 環境
    # render_mode="human" for visualization, None for faster training
    env = gym.make("CartPole-v1") #, render_mode="human") 

    # 获取状态和动作空间的维度
    state_tuple = env.reset()
    state: np.ndarray = state_tuple[0] # env.reset() now returns a tuple (obs, info)
    state_dim: int = len(state)
    action_dim: int = env.action_space.n

    # 初始化 DQN 代理
    # Batch size reduced to 64 as in the original script's main block
    agent = DQNAgent(state_dim, action_dim, batch_size=64, update_every=4, lr=1e-3) 

    # 训练参数
    num_episodes: int = 600
    episode_durations: List[int] = []
    rewards: List[float] = []
    avg_losses: List[float] = [] # Store average loss per episode

    print(f"Starting training for {num_episodes} episodes...")

    for i_episode in range(num_episodes):
        episode_start_time = time.time()
        
        # 初始化环境和状态
        state_tuple = env.reset()
        state: np.ndarray = state_tuple[0] # state is already a np.float32 array

        total_reward: float = 0
        current_episode_losses: List[float] = []
        terminated: bool = False
        truncated: bool = False
        t: int = 0 # Timesteps in current episode
        
        while not terminated and not truncated:
            # 选择并执行动作
            action_tensor: torch.Tensor = agent.select_action(state)
            action: int = action_tensor.item() # Convert tensor action to int for env.step
            
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Done flag for ReplayMemory
            done: bool = terminated or truncated
            
            # 处理下一个状态 (observation is already np.float32 array)
            next_state: Optional[np.ndarray] = None if terminated else observation

            # 将转换存储在经验回放缓冲区中
            agent.memory.store(state, action, next_state, reward, done)

            # 移动到下一个状态
            # If terminated, next_state is None, loop will break.
            # If not terminated, state becomes the new observation.
            if next_state is not None:
                state = next_state
            else: # Handle the case where next_state is None (episode ended)
                # state = observation # Not strictly necessary to update state if episode ended
                pass


            # 执行一步优化
            loss_item: Optional[float] = agent.learn()
            if loss_item is not None:
                current_episode_losses.append(loss_item)

            t += 1
            if done:
                episode_durations.append(t) # Duration is number of steps taken
                rewards.append(total_reward)
                if current_episode_losses: # Avoid division by zero if no learning happened
                    avg_losses.append(sum(current_episode_losses) / len(current_episode_losses))
                else:
                    # Could append a placeholder if no learning steps occurred, or skip
                    # If avg_losses can be empty, the print statement needs to handle it.
                    pass 
                
                if (i_episode + 1) % 50 == 0:
                    episode_run_time = time.time() - episode_start_time
                    # ***** FIXED LINE BELOW *****
                    print(f"Episode {i_episode+1}/{num_episodes} | Duration: {t} | "
                          f"Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | "
                          f"Avg Loss: {(f'{avg_losses[-1]:.4f}' if avg_losses else 'N/A')} | Time: {episode_run_time:.2f}s")
                break
        # End of episode while loop
    # End of training loop

    total_training_time = time.time() - start_time
    print(f'\nTraining complete. Total time: {total_training_time:.2f} seconds')
    env.close()

    # 绘图代码来可视化训练结果
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # Use subplots for better control
    
    axs[0].plot(episode_durations)
    axs[0].set_title('Episode Durations')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Duration (timesteps)')
    
    axs[1].plot(rewards)
    axs[1].set_title('Episode Rewards')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Total Reward')
    
    if avg_losses:
        axs[2].plot(avg_losses)
        axs[2].set_title('Average Training Loss per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Average Loss')
    else:
        axs[2].set_title('Training Loss (No data)')
        axs[2].text(0.5, 0.5, 'No learning steps recorded', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

    
    plt.tight_layout()
    plt.show()