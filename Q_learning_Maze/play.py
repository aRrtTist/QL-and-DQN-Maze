"""
游戏的主程序，调用机器人的 Q learning 决策大脑 和 Maze 环境
"""

from env import Maze
from q_learning import QLearning


def update():
    for episode in range(100):
        # 初始化 state（状态）
        state = env.reset()

        step_count = 0  # 记录走过的步数

        while True:
            # 更新可视化环境
            env.render()

            # RL 大脑根据 state 挑选 action
            action = RL.choose_action(str(state))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state, reward 和 done (是否是踩到炸弹或者找到宝藏)
            state_, reward, done = env.step(action)

            step_count += 1  # 增加步数

            # 机器人大脑从这个过渡（transition） (state, action, reward, state_) 中学习
            RL.learn(str(state), action, reward, str(state_))

            # 机器人移动到下一个 state
            state = state_

            # 如果踩到炸弹或者找到宝藏, 回合结束
            if done:
                print("回合 {} 结束. 总步数 : {}\n".format(episode+1, step_count))
                break

    # 结束游戏并关闭窗口
    print('游戏结束')
    env.destroy()


if __name__ == "__main__":
    # 创建环境 env 和 RL
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)))

    # 开始可视化环境
    env.after(100, update)
    env.mainloop()

    print('\nQ 表:')
    print(RL.q_table)
