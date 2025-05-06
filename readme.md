經典深度Q網絡(DQN)架構，主要特徵如下：

# 網絡結構
	* 三層全連接神經網絡 (輸入層 → 128隱藏層 → 128隱藏層 → 輸出層)
	* 使用ReLU激活函數
	* 輸出層直接映射到動作空間
	
	
# 特殊實現細節
	* 採用雙網絡設計（策略網絡+目標網絡）
	* 使用deque實現循環緩衝區
	* 批次處理 (Batch Processing) 
		* 整批一起轉換和計算 (torch.tensor(np.array(batch.state)), torch.cat(batch.action))，而不是用迴圈一個一個處理。
		* 充分利用了 PyTorch 和 NumPy 底層的向量化運算能力，速度遠快於逐一處理。
	* 軟更新(Soft Update)而非硬更新(Hard Update)
		* 除了主要的策略網路 (policy_net)，還額外建立了一個結構相同但參數更新較慢的「目標網路」 (target_net)。
		* 計算目標 Q 值時使用這個目標網路。目標網路的參數不是直接複製，而是緩慢地（由 tau 控制）從策略網路更新過來（稱為軟更新）。
		* 這樣做可以讓學習的目標值不會變動得太劇烈，像加了穩定器一樣，有助於訓練過程的穩定和收斂。
	
# 優化點
	* 使用ε-greedy探索策略與漸進衰減機制: ε = ε_end + (ε_start - ε_end) * exp(-steps/ε_decay)
	* 採用Smooth L1 Loss(Huber Loss): 相較於均方誤差 (MSE)，它對於異常值（例如突然出現的超大獎勵或懲罰）比較不敏感，可以讓訓練過程更穩定，不容易被極端值帶偏。
	* AdamW優化器(Adam 優化器的改良版)搭配AMSGrad
	* 實施經驗回放(Experience Replay)技術: 儲存過去的經驗（狀態、動作、獎勵等）。deque 是一種雙向佇列，在固定容量下，新增資料和移除最舊資料的效率都很高。
	* 應用梯度裁剪(Clip Value=100)防止梯度爆炸

此實現針對CartPole控制問題進行訓練，符合2013年Mnih等人提出的經典DQN架構。

CartPole控制問題: 車上有根直立的棍子，如何在行進時保持棍子平衡

https://www.kaggle.com/code/shvara/qdn-test

# 改進
	* GPU 加速
	* 計算下一狀態的價值時，會先把已經結束的狀態 (done=True) 過濾掉 (non_final_mask, non_final_next_states)，避免不必要的計算。
	* 降低更新頻率 (Update Frequency):透過 update_every 參數，可以設定每隔幾步才進行一次 learn() 操作和目標網路的更新。這可以減少總體的計算量，雖然可能會稍微延遲學習的反應，但在計算資源有限時是一種權衡。
