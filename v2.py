import numpy as np


def Viterbi(A, B, PI, V, Q, obs):
    N = len(Q)
    T = len(obs)
    delta = np.array([[0] * N] * T, dtype=np.float64)
    phi = np.array([[0] * N] * T, dtype=np.int64)
    # 初始化
    for i in range(N):
        delta[0, i] = PI[i] * B[i][V.index(obs[0])]
        phi[0, i] = 0

    # 递归计算
    for i in range(1, T):
        for j in range(N):
            tmp = [delta[i - 1, k] * A[k][j] for k in range(N)]
            delta[i, j] = max(tmp) * B[j][V.index(obs[i])]
            phi[i, j] = tmp.index(max(tmp))

    # 最终的概率及节点
    P = max(delta[T - 1, :])
    I = int(np.argmax(delta[T - 1, :]))

    # 最优路径path
    path = [I]
    for i in reversed(range(1, T)):
        end = path[-1]
        path.append(phi[i, end])

    hidden_states = [Q[i] for i in reversed(path)]

    return P, hidden_states


def viterbe_2(trans_p, emit_p, start_p, obs, states,  obs_seq):
    """
    t时刻观测状态是T，则t时刻所有情况计算：
    :param trans_p:
    :param emit_p:
    :param start_p:
    :param states: 隐状态集合
    :param obs: 观测集合
    :param obs_seq:
    :return:
    """
    N = len(states)  # 隐状态长度
    T = len(obs_seq)  # 实际观测序列长度

    # delta[i][j]表示从头转移到(i,j)的概率
    delta = np.array([[0] * N] * T, dtype=np.float64)
    # 记录上一节点状态（也就是上一轮转到这一轮的最大概率的节点）
    phi = np.array([[0] * N] * T, dtype=np.int64)

    # 初始化, 第一行概率 = 初始概率*发射到obs_seq[0]
    for i in range(N):
        delta[0, i] = start_p[i] * emit_p[i][obs.index(obs_seq[0])]
        phi[0, i] = 0

    # 更新
    # i表示每个时刻
    for i in range(1, T):
        for j in range(N):
            # 对上一轮所有隐状态（N个），到达节点k概率（delta[i - 1][k]） * 节点k转移到j的发射概率（trans_p[k][j]）
            temp = [delta[i - 1][k] * trans_p[k][j] for k in range(N)]
            # 选取从上一轮到节点j最大的概率 * 节点j到i时刻观察状态的转移概率
            delta[i][j] = max(temp) * emit_p[j][obs.index(obs_seq[i])]
            # 记录下上一轮最大转到i轮最大概率的节点下表
            phi[i][j] = temp.index(max(temp))

    # 最终的概率和最优节点
    P = max(delta[T - 1, :])
    I = int(np.argmax(delta[T - 1, :]))

    # 最优路径path
    path = [I]
    for i in reversed(range(1, T)):
        end = path[-1]
        path.append(phi[i, end])
    hidden_states = [states[i] for i in reversed(path)]
    return P, hidden_states


def main():
    # 状态集合
    Q = ('欢乐谷', '迪士尼', '外滩')
    # 观测集合
    V = ['购物', '不购物']
    # 转移概率: Q -> Q
    A = [[0.8, 0.05, 0.15],
         [0.2, 0.6, 0.2],
         [0.2, 0.3, 0.5]
         ]

    # 发射概率, Q -> V
    B = [[0.1, 0.9],
         [0.8, 0.2],
         [0.3, 0.7]
         ]

    # 初始概率
    PI = [1 / 3, 1 / 3, 1 / 3]

    # 观测序列
    obs = ['不购物', '购物', '购物']

    P, hidden_states = viterbe_2(A, B, PI, V, Q, obs)
    print('最大的概率为: %.5f.' % P)
    print('隐藏序列为：%s.' % hidden_states)



main()
