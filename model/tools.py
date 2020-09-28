import numpy as np


def _trans(Q, X):
    """
    计算两个矩阵的欧氏距离：
    (Q-X)^2 = Q^2 + X^2 - 2*Q*X

    :param Q: shape(m,d)
    :param X: shape(n,d)
    :return: shape(m,n)
    """

    # 计算维度
    row_q, col_q = Q.shape
    row_x, col_x = X.shape
    if col_q != col_x:
        raise RuntimeError('colx must be equal with coly')
    # 计算 2*Q*X，结果：(m,n)
    qx = np.dot(Q, X.T)
    # 计算Q^2，结果：(m,n)
    qq = np.repeat(np.reshape(np.sum(Q * Q, axis=1), (row_q, 1)), repeats=row_x, axis=1)
    # 计算X^2, 结果: (m,n)
    xx = np.repeat(np.reshape(np.sum(X * X, axis=1), (row_x, 1)), repeats=row_q, axis=1).T

    res = qq + xx - 2 * qx
    return res


if __name__ == '__main__':
    Q = np.full((2, 5), 2)
    print("Q:")
    print(Q)
    X = np.full((4, 5), 1)
    print("X:")
    print(X)
    _trans(Q, X)
