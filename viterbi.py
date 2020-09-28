import numpy as np

states = ('Rainy', 'Sunny')  # 隐状态数量

# 观测集合
obs = ['walk', 'shop', 'clean']

observations = ('walk', 'shop', 'clean')  # 可见序列

start_probability = {'Rainy': 0.6, 'Sunny': 0.4}  # 先验概率（初始概率）

transition_probability = {
    'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
}  # 隐状态转换概率

emission_probability = {
    'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}  # 输出概率（隐状态->可见状态）


# 计算最可能的天气序列


def print_dptable(V):
    for i in range(len(V)):
        print("%7d" % i)
    print()
    for y in V[0].keys():
        print("%.5s: " % y)
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]))
        print()


def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    动态规划：
    初始化：
    转移函数：
    dp[i] = dp[j]

    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率 （隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}

    # 初始化初始状态 (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            # 概率 隐状态 = 前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            temp = [(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states]
            (prob, state) = max(temp)
            # 记录最大概率
            V[t][y] = prob
            # 记录路径
            newpath[y] = path[state] + [y]

        # 不需要保留旧路径
        path = newpath

    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])


def viterbe_2(trans_p, emit_p, start_p, states, obs, obs_seq):
    """
    t时刻观测状态是T，则t时刻所有情况计算：
    for i in 隐状态集合states：
        for j in 观测集合obs：
            前一个状态是x的概率 * 状态x转移到状态y的概率 * 状态y下进行T行为的概率
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
    delta = np.array([[0] * N] * T, dtype=np.float64)
    phi = np.array([[0] * N] * T, dtype=np.int64)

    # 初始化
    for i in range(N):
        delta[0, i] = start_p[i] * emit_p[i][obs.index(obs_seq[0])]
        phi[0, i] = 0
    # 更新
    for i in range(1, T):
        for j in range(N):
            temp = [delta[i - 1][k] * trans_p[k][k] for k in range(N)]
            delta[i][j] = max(temp) * emit_p[j][obs.index(obs_seq[i])]
            phi[i][j] = temp.index(max(temp))

    # 最终概率和节点
    P = max(delta[T - 1, :])
    I = int(np.argmax(delta[T - 1, :]))

    # 最优路径path
    path = [I]
    for i in reversed(range(1, T)):
        end = path[-1]
        path.append(phi[i, end])
    hidden_states = [states[i] for i in reversed(path)]
    return P, hidden_states


if __name__ == '__main__':
    res = viterbi(observations,
                  states,
                  start_probability,
                  transition_probability,
                  emission_probability)
    print(res)
