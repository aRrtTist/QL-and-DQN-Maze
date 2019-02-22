"""
Q Learning 算法。做决策的部分，相当于机器人的大脑
"""

import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, e_greedy=0.1):
        self.actions = actions        # action 列表
        self.lr = learning_rate       # 学习速率
        self.gamma = discount_factor  # 折扣因子
        self.epsilon = e_greedy       # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)  # Q 表

    # 如果还没有当前 state, 就插入一组全 0 数据, 作为这个 state 的所有 action 的初始值
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 插入一组全 0 数据
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 根据 state 来选择 action
    def choose_action(self, state):
        self.check_state_exist(state)  # 检测此 state 是否在 q_table 中存在
        # 选行为，用 Epsilon Greedy 贪婪方法
        if np.random.uniform() < self.epsilon:
            # 随机选择 action
            action = np.random.choice(self.actions)
        else:  # 选择 Q 值最高的 action
            state_action = self.q_table.loc[state, :]
            # 同一个 state, 可能会有多个相同的 Q action 值, 乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        return action

    # 学习。更新 Q 表中的值
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  # 检测 q_table 中是否存在 s_

        q_predict = self.q_table.loc[s, a]  # 根据 Q 表得到的 估计（predict）值

        # q_target 是现实值
        if s_ != 'terminal':  # 下个 state 不是 终止符
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # 下个 state 是 终止符

        # 更新 Q 表中 state-action 的值
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

