"""
用 Deep Q Learning 玩迷宫（Maze）游戏
"""

from env import Maze
from deep_q_learning import DeepQLearning


def update():
    step = 0  # 控制什么时候学习
    
    for episode in range(300):
        # 初始化 state（状态）
        state = env.reset()

        step_count = 0  # 记录走过的步数

        while True:
            # 更新可视化环境
            env.render()

            #  大脑根据 state 挑选 action
            action = dqn.choose_action(state)

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state, reward 和 done (是否是踩到炸弹或者找到宝藏)
            state_, reward, done = env.step(action)

            step_count += 1  # 增加步数

            # DeepQLearning 大脑存储这个过渡（transition） (state, action, reward, state_) 的记忆
            dqn.store_transition(state, action, reward, state_)

            # 控制学习起始时间和频率 (先积累一定记忆才开始学习)
            if (step > 200) and (step % 5 == 0):
                dqn.learn()

            # 机器人移动到下一个 state
            state = state_

            # 如果踩到炸弹或者找到宝藏, 回合结束
            if done:
                print("回合 {} 结束. 总步数 : {}\n".format(episode + 1, step_count))
                break

            step += 1  # 总步数

    # 结束游戏并关闭窗口
    print('游戏结束.\n')
    env.destroy()


if __name__ == "__main__":
    # 创建游戏环境
    env = Maze()

    output_graph_boolean = True  # 是否输出 Tensorboard 日志文件
    # 创建 DeepQLearning 对象
    dqn = DeepQLearning(env.n_actions, env.n_features,
                        learning_rate=0.01,
                        discount_factor=0.9,
                        e_greedy=0.1,
                        replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                        memory_size=2000,  # 记忆上限
                        output_graph=output_graph_boolean
                        )

    # 开始可视化环境
    env.after(100, update)
    env.mainloop()

    if output_graph_boolean:
        print('神经网络的日志文件生成在 logs 文件夹里，请用以下命令在 TensorBoard 中查看网络模型图：')
        print('\ttensorboard --logdir=logs\n')

