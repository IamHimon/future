import math
from collections import Counter, defaultdict
from functools import cmp_to_key
from typing import List

from MonotonicQueue import MonotonicQueue
from TreeNode import TreeNode, BTree
from base_classes import ListNode


def maxSlidingWindow_239(nums: List[int], k: int) -> List[int]:
    """
    暴力解法，超时
    """
    res = []
    res.append(max(nums[:k]))
    for i in range(1, len(nums) - k + 1):
        if nums[i + k - 1] < res[-1]:
            res.append(max(nums[i:i + k]))
        else:
            res.append(nums[i + k - 1])
    return res


def maxSlidingWindow_239_1(nums: List[int], k: int) -> List[int]:
    """
    使用单调队列实现
    """
    mono_queue = MonotonicQueue()
    res = []
    if k == 1:
        return nums
    # 先处理前k个
    _max = max(nums[:k])
    res.append(_max)
    for item in nums[:k]:
        mono_queue.push(item)

    for i in range(k, len(nums)):
        # 单调栈中删除窗口前面一个元素
        mono_queue.pop(nums[i - k])
        mono_queue.push(nums[i])
        res.append(mono_queue.max())

    return res


def lowestCommonAncestor_236(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    后续遍历的思路
    """
    if not root:
        return None
    if root or root == p or root == q:
        return root
    left = lowestCommonAncestor_236(root.left, p, q)
    right = lowestCommonAncestor_236(root.right, p, q)
    # 如果left和right都非空，说明p和q在一左一右
    if left and right:
        return root
    # 如果left非空，right为空，说明p和q都在左边找到，右边什么都没有
    if left:
        return left
    # 同理
    if right:
        return right


def binaryTreePaths_257(root: TreeNode) -> List[str]:
    if not root:
        return []
    result = []

    def _dfs(root, path):
        if not root:
            return
        if not root.right and not root.left:
            path += str(root.val)
            result.append(path)
            return

        _dfs(root.left, path + str(root.val) + '->')
        _dfs(root.right, path + str(root.val) + '->')

    _dfs(root, '')
    return result


def exist_offer12(board: List[List[str]], word: str) -> bool:
    if not board:
        return False
    L = len(board[0])
    H = len(board)

    def _dfs(board, h, l, word):
        # word空了，说明全部匹配到，返回true
        if not word:
            return True
        # 如果索引越界，或者值不匹配，返回false
        if h < 0 or h >= H or l < 0 or l >= L or board[h][l] != word[0]:
            return False
        temp = board[h][l]  # 暂存（h，l）位置的值
        board[h][l] = '\0'
        if _dfs(board, h - 1, l, word[1:]) or _dfs(board, h + 1, l, word[1:]) \
                or _dfs(board, h, l - 1, word[1:]) or _dfs(board, h, l + 1, word[1:]):
            return True
        board[h][l] = temp  # （h，l）位置的值回复
        return False

    for h in range(H):
        for l in range(L):
            if _dfs(board, h, l, word):
                return True
    return False


def cuttingRope_offer14(n: int) -> int:
    """
    回溯方法，超时
    """
    if not n:
        return 0
    result = []

    def _method(n, mul_res, m):
        if n <= 1:
            result.append(mul_res)
            return
        for i in range(1, n + 1):
            if n - i <= 0 and m == 0:
                continue
            _method(n - i, mul_res * i, m + 1)
        return

    _method(n, 1, 0)
    print(result)
    return max(result)


def cuttingRope_offer14_1(n: int) -> int:
    """
    动态规划解决。
    假设第一个分出来的正整数是k，则剩下的就是n-k。这是，最大乘积，就是k拆分之后的最大乘积*n-k拆分的最大乘积，存在递推关系，可以使用动态规划解决。
    定义：
    dp[i]表示将正整数i拆分成至少两个正整数之后，这些正整数的最大的乘积。
    所以，问题变成，如何确定dp[i]的值，这里需要对i继续进行拆分判断。
    方程：
    假设对正整数i拆分,当第一个整数是j：
    dp[i] = max(j*(i-j), j*dp[i-j]， dp[i])
    j*i-j:把i分成两段，一段j一段i-j
    j*dp[i-j]：分成j之后，把i-j再继续拆分
    初始化：
    dp[0]=0
    dp[1]=0
    dp[2]=1
    """
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        # 拆解i，确定最大的dp[i]
        for j in range(1, i):
            dp[i] = max(j * (i - j), j * dp[i - j], dp[i])
    return dp[n]


def maxProfit_121(prices: List[int]) -> int:
    if not prices:
        return 0
    max_pro = 0
    min_buy = prices[0]
    for price in prices[1:]:
        min_buy = min(min_buy, price)
        max_pro = max(max_pro, price - min_buy)
    return max_pro


def ladderLength_127(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    BFS
    """
    # 建立通用neighbours
    L, neighbours_dict = len(beginWord), defaultdict(list)
    mask = {}
    for w in wordList:
        mask[w] = 0
        for _ in range(L):
            neighbours_dict[w[:_] + "*" + w[_ + 1:]].append(w)

    queue = [beginWord]
    level = 1
    while queue:
        level += 1
        size = len(queue)
        while size:
            cur = queue.pop(0)
            size -= 1
            for i in range(L):
                _cur = cur[:i] + '*' + cur[i + 1:]
                if _cur not in neighbours_dict:
                    continue
                for aim in neighbours_dict.get(_cur):
                    if mask[aim] == 1:
                        continue
                    if aim == endWord:
                        return level
                    mask[aim] = 1
                    queue.append(aim)
    return 0


if __name__ == '__main__':
    # tree1 = BTree()
    # tree1.create_btree([1, 2, 3, None, 5])
    # #
    # n1 = ListNode(1)
    # n2 = ListNode(2)
    # n3 = ListNode(3)
    # n4 = ListNode(4)
    # n5 = ListNode(5)
    # n6 = ListNode(6)
    # # head.next = n1
    # n1.next = n2
    # n2.next = n3
    # n3.next = n4
    # n4.next = n5
    # n5.next = n6
    #
    # nums = [["1", "0", "1", "1", "1"], ["1", "0", "1", "0", "1"], ["1", "1", "1", "0", "1"]]
    # res = findKthLargest([7,6,5,4,3,2,1],2)
    # res1 = computeArea(0, 0, 0, 0, -1, -1, 1, 1)
    # res1 = computeArea(-3, -3, 3, -1, -2, -2, 2, 2)
    # res1 = computeArea(-2, -2, 2, 2, 1, 1, 3, 3)
    # res1 = findOrder_210(4, [[1,0],[2,0],[3,1],[3,2]])
    # res1 = computeArea_223(-3, -3, 3, 3, -3, -3, 3, 3)
    # print(res)
    # print(res1)

    # Your WordDictionary object will be instantiated and called as such:
    # obj = WordDictionary()
    # obj.addWord("bad")
    # obj.addWord("dad")
    # obj.addWord("dadqw")
    # res = obj.search("b..")
    # print(res)
    # m = [
    #     [1, 4, 7, 11, 15],
    #     [2, 5, 8, 12, 19],
    #     [3, 6, 9, 16, 22],
    #     [10, 13, 14, 17, 24],
    #     [18, 21, 23, 26, 30]
    # ]

    result = ladderLength_127("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    print(result)
