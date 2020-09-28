import math
from collections import Counter
from functools import cmp_to_key
from typing import List

from TreeNode import TreeNode, BTree
from base_classes import ListNode


def largestNumber_179(nums: List[int]) -> str:
    def compare(v1, v2):
        v1_str = str(v1)
        v2_str = str(v2)
        return v1_str + v2_str > v2_str + v1_str

    def _partition(left, right, nums):
        pivot_value = nums[left]
        while left < right:
            # 从右边找第一个小于pivot的位置, 找到并交换到left位置上
            while left < right and not compare(nums[right], pivot_value):
                right -= 1
            nums[left] = nums[right]
            # 从左边找第一个大于pivot的位置,找到并交换到right位置上
            while left < right and not compare(pivot_value, nums[left]):
                left += 1
            nums[right] = nums[left]
        # 分割点pivot赋值到left位置
        nums[left] = pivot_value
        return left

    def _ite(left, right, nums):
        if left >= right:
            return
        mid = _partition(left, right, nums)
        _ite(left, mid - 1, nums)
        _ite(mid + 1, right, nums)

    _ite(0, len(nums) - 1, nums)
    if nums[0] == 0:
        return '0'
    res = ''
    for num in nums:
        res += str(num)
    return res


def largestNumber_179_1(nums: List[int]) -> str:
    if not nums:
        return ''
    nums = map(str, nums)
    key = cmp_to_key(lambda x, y: int(y + x) - int(x + y))
    s_nums = sorted(nums, key=key)
    return ''.join(s_nums).lstrip('0') or '0'


def findRepeatedDnaSequences_187(s: str) -> List[str]:
    if not s:
        return []
    temp = {}
    result = set()
    for i in range(len(s) - 9):
        sub_seq = s[i:i + 10]
        if sub_seq in temp:
            temp[sub_seq] += 1
            result.add(sub_seq)
        else:
            temp[sub_seq] = 1
    return list(result)


def findRepeatedDnaSequences_187_1(s: str) -> List[str]:
    def _cal_hash1(sub_ints):
        res = 0
        for j in range(10):
            res += sub_ints[j] * math.pow(4, 10 - j - 1)
        return res

    def _cal_hash(sub_ints):
        h = 0
        for j in range(10):
            h = h * 4 + sub_ints[j]
        return h

    aL = math.pow(4, 10)
    to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ints = [to_int[c] for c in s]
    h = 0
    result = set()
    temp = set()
    for i in range(len(ints) - 9):
        if i == 0:
            h = _cal_hash(ints[0:10])
        else:
            h = h * 4 - ints[i - 1] * aL + ints[i + 9]
        if h in temp:
            result.add(s[i:i + 10])
        temp.add(h)
    return list(result)


def reverseBits_190(n: int) -> int:
    def _to_bit(n):
        # trans to binary
        bin_str = bin(n).replace('0b', '')
        # padding
        bin_str = '0' * (32 - len(bin_str)) + bin_str
        re_bin_str = ''.join(reversed(bin_str))
        return re_bin_str

    def _from_bit(bits):
        res = int(bits, 2)
        return res

    bits = _to_bit(n)
    return _from_bit(bits)


def hammingWeight_191(n: int) -> int:
    bits = bin(n).replace('0b', '')
    bits = [int(b) for b in bits]
    return sum(bits)


def hammingWeight_191_1(n: int) -> int:
    """
    n 与 n-1做与操作，会消除n中的最后一个1
    比如：n:      0100 1100
        n-1:     0100 1011
        n & n-1: 0100 1000
    :param n:
    :return:
    """
    res = 0
    while n:
        res += 1
        n &= (n - 1)
    return res


def rightSideView(root: TreeNode) -> List[int]:
    if not root:
        return []
    res = [root.val]

    def _level(root):
        queue = [root]
        while queue:
            temp_queue = []
            while queue:
                top = queue.pop(0)
                if top.left:
                    temp_queue.append(top.left)
                if top.right:
                    temp_queue.append(top.right)
            if temp_queue:
                res.append(temp_queue[-1].val)
            queue = temp_queue

    _level(root)
    return res


def rightSideView_2(root: TreeNode) -> List[int]:
    if not root:
        return []
    result = []

    def _level(root, level):
        if not root:
            return
        if len(result) <= level:
            result.append(0)
        result[level] = root.val
        if root.left:
            _level(root.left, level + 1)
        if root.right:
            _level(root.right, level + 1)

    _level(root, 0)
    return result


def numIslands_200(grid: List[List[str]]) -> int:
    """

    模仿层次遍历，每层遍历，每一个出栈的节点向上、下、左、右 四个方向扩展
    :param grid:
    :return:
    """
    if not grid:
        return 0
    H = len(grid)
    L = len(grid[0])
    flag = [[0] * L for i in range(H)]

    h = 0
    res = 0
    while h < H:
        l = 0
        while l < L:
            if grid[h][l] == '0':
                flag[h][l] = 1
                l += 1
                continue
            if flag[h][l]:
                l += 1
                continue
            stack = []
            if grid[h][l] == '1':
                flag[h][l] = 1
                stack.append((h, l))
                res += 1

            # 右边进栈
            while l + 1 < L and grid[h][l + 1] == '1':
                stack.append((h, l + 1))
                flag[h][l + 1] = 1
                l += 1
            print(stack)
            # 下面入栈
            while stack:
                temp_stack = []
                while stack:
                    (_h, _l) = stack.pop(0)

                    if _h - 1 >= 0 and grid[_h - 1][_l] == '1' and not flag[_h - 1][_l]:
                        # 上面进栈
                        temp_stack.append((_h - 1, _l))
                        flag[_h - 1][_l] = 1
                        # 右边进栈
                        right_l = _l
                        while right_l + 1 < L and grid[_h - 1][right_l + 1] == '1' and not flag[_h - 1][right_l + 1]:
                            temp_stack.append((_h - 1, right_l + 1))
                            flag[_h - 1][right_l + 1] = 1
                            right_l += 1
                        # 左边进栈
                        left_l = _l
                        while left_l - 1 >= 0 and grid[_h - 1][left_l - 1] == '1' and not flag[_h - 1][left_l - 1]:
                            temp_stack.append((_h - 1, left_l - 1))
                            flag[_h - 1][left_l - 1] = 1
                            left_l -= 1

                    if _h + 1 < H and grid[_h + 1][_l] == '1' and not flag[_h + 1][_l]:
                        # 下面进栈
                        temp_stack.append((_h + 1, _l))
                        flag[_h + 1][_l] = 1
                        # 右边进栈
                        right_l = _l
                        while right_l + 1 < L and grid[_h + 1][right_l + 1] == '1' and not flag[_h + 1][right_l + 1]:
                            temp_stack.append((_h + 1, right_l + 1))
                            flag[_h + 1][right_l + 1] = 1
                            right_l += 1
                        # 左边进栈
                        left_l = _l
                        while left_l - 1 >= 0 and grid[_h + 1][left_l - 1] == '1' and not flag[_h + 1][left_l - 1]:
                            temp_stack.append((_h + 1, left_l - 1))
                            flag[_h + 1][left_l - 1] = 1
                            left_l -= 1
                stack = temp_stack
            l += 1
        h += 1

    return res


def getPermutation_60(n: int, k: int) -> str:
    if not k:
        return ''

    nums = [i for i in range(1, n + 1)]
    result = []

    def _func(nums, k, res):
        new_n = len(nums)
        # 提前结束
        if k == 1:
            for n in nums:
                result.append(n)
                res += str(n)
            return res
        # 计算new_n-1阶乘
        t_k = 1
        for i_k in range(1, new_n):
            t_k *= i_k
        # 确定取数位置
        div = (k - 1) // t_k
        mod = (k - 1) % t_k + 1

        # 当前数字取出来，然后从nums中移除
        chose = nums[div]
        result.append(chose)
        res += str(chose)
        nums.remove(chose)
        return _func(nums, mod, res)

    res = _func(nums, k, '')
    return res


def isSameTree_100(p: TreeNode, q: TreeNode) -> bool:
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree_100(p.left, q.left) and isSameTree_100(p.right, q.right)


def isSameTree_100_1(p: TreeNode, q: TreeNode) -> bool:
    stack_p = [p]
    stack_q = [q]
    while stack_p and stack_q:
        top_p = stack_p.pop()
        top_q = stack_q.pop()
        if not top_p and not top_q:
            continue
        # 　一直往左边,同时遇到右节点入栈
        if not top_q or not top_p or top_q.val != top_p.val:
            return False

        stack_p.append(top_p.right)
        stack_q.append(top_q.right)
        stack_q.append(top_q.left)
        stack_p.append(top_p.left)

    return True


def isSymmetric_101(root: TreeNode) -> bool:
    """
    # 层次遍历，判断每一层的结果是不是回文串
    :param root:
    :return:
    """
    if not root:
        return True

    def _is_sym(nums):
        left = 0
        right = len(nums) - 1
        while left <= right:
            if nums[left] != nums[right]:
                return False
            left += 1
            right -= 1
        return True

    queue = [root]
    while queue:
        temp_queue = []
        level_nums = []
        while queue:
            top = queue.pop(0)
            if top.left:
                temp_queue.append(top.left)
                level_nums.append(top.left.val)
            else:
                level_nums.append(None)
            if top.right:
                temp_queue.append(top.right)
                level_nums.append(top.right.val)
            else:
                level_nums.append(None)
        if temp_queue:
            if not _is_sym(level_nums):
                return False
        queue = temp_queue
    return True


def isSymmetric_101_1(root: TreeNode) -> bool:
    """
    1. 判断当前两个节点是否对称（val相等）
    2. 判断左节点的右节点与右节点的左节点是否对称
    3. 判断左节点的左节点与右节点的右节点是否对称

    :param root:
    :return:
    """
    if not root:
        return True

    def _checke(left, right):
        if not left and not right:
            return True
        if not left or not right or left.val != right.val:
            return False
        return _checke(left.left, right.right) and _checke(left.right, right.left)

    return _checke(root.left, root.right)


def maxDepth_104(root: TreeNode) -> int:
    if not root:
        return 0
    return max(1 + maxDepth_104(root.left), 1 + maxDepth_104(root.right))


def isBalanced_110(root: TreeNode) -> bool:
    """
    自顶向下：遍历，判断每个节点的左右子树高度，比较是否平衡
    :param root:
    :return:
    """

    def _get_length(root):
        if not root:
            return 0
        return max(1 + _get_length(root.left), 1 + _get_length(root.right))

    if not root:
        return True
    l_high = _get_length(root.left)
    r_high = _get_length(root.right)
    print(root.val)
    print("l_h:%s, r_h:%s" % (l_high, r_high))
    if math.fabs(l_high - r_high) > 1:
        return False
    return isBalanced_110(root.left) and isBalanced_110(root.right)


def isBalanced_110_1(root: TreeNode) -> bool:
    """
    自底向上（提前阻断）：
    同时判断是否平衡，以及进行更新高度
    :param root:
    :return:
    """

    def _judeg(root):
        if not root:
            return True, 0
        left_is_bal, left_h = _judeg(root.left)
        if not left_is_bal:
            return False, 0
        right_is_bal, right_h = _judeg(root.right)
        if not right_is_bal:
            return False, 0
        return math.fabs(left_h - right_h) <= 1, 1 + max(left_h, right_h)

    res, _ = _judeg(root)
    return res


def maxProfit_121(prices: List[int]) -> int:
    """
    暴力法
    :param prices:
    :return:
    """
    max_profit = 0
    for i in range(len(prices) - 1):
        max_price = max(prices[i + 1:])
        max_profit = max(max_price - prices[i], max_profit)
    return max_profit


def maxProfit_121_1(prices: List[int]) -> int:
    """
    动态规划
    定义：
    dp[i] ,(前i天最大收益，前i天最小买入价格)
    初始化：
    dp[0] = (0, prices[0])
    方程：
    前i天的最大收益 = max(前i-1天最大收益，i天价格-前i-1天最小买入价格)
    dp[i][0] = max{dp[i](0), price[i]-dp[i][1]}
    :param prices:
    :return:
    """
    if not prices:
        return 0
    _min = prices[0]
    _max = 0
    for i in range(1, len(prices)):
        _max = max(prices[i] - _min, _max)
        _min = min(prices[i], _min)
    return _max


def maxProfit_122(prices: List[int]) -> int:
    """
    遍历数组，只要后面比前面大，就加起来
    :param prices:
    :return:
    """
    res = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            res += (prices[i] - prices[i - 1])
    return res


def hasCycle(head: ListNode, n) -> bool:
    """
    判断连表是否有环：
    使用快慢指针，快指针走一步，慢指针走两步，如果相遇则肯定有环，否则无环
    征明为什么一定一快一慢会相遇：
        用两个移动的指针可以确保：如果有环的话两个指针都会进入有环的部分。
        一旦进入有环的部分，一快一慢，就相当于一个静止另一个移动。
    :param head:
    :param n:
    :return:
    """
    if not head:
        return False
    below = head
    ahead = head
    while below and ahead.next:
        below = below.next
        ahead = ahead.next.next
        if below == ahead:
            return True
    return False


def canCompleteCircuit_134(gas: List[int], cost: List[int]) -> int:
    L = len(gas)
    for i in range(L):
        # 如果起点的cost大于gas，肯定不行的
        if cost[i] > gas[i]:
            continue
        # 循环执行
        tank = 0
        pos = i
        for j in range(i, L + i + 1):
            # 循环执行
            if j >= L:
                pos = j - L
            else:
                pos = j
            # print(pos)
            # 先加油，再减去预计消耗
            tank = tank + gas[pos] - cost[pos]
            # 如果油箱减去消耗小于0了，就达不到下一个加油站了，则结束这一轮
            if tank < 0:
                break
        # 如果回到原点，则成功
        if pos == i:
            return True
    return False


def countPrimes_204(n: int) -> int:
    """
    超时，记下每个素数
    :param n:
    :return:
    """

    def _judge(num, primes):
        for pri in primes:
            if num % pri == 0:
                return False
        return True

    if n <= 1:
        return 0
    count = 0
    before_primes = []
    for i in range(2, n):
        if _judge(i, before_primes):
            count += 1
            before_primes.append(i)
    return count


def countPrimes1_204(n: int) -> int:
    """
    排除法,
    :param n:
    :return:
    """
    if n <= 1:
        return 0
    # i位置是否是质数,1：是
    flag = [1] * n
    flag[0] = 0
    flag[1] = 1

    count = 0
    for i in range(2, n):
        if flag[i]:
            count += 1
            # 排除
            j = i
            cur = i
            while cur * j < n:
                flag[cur * j] = 0
                j += 1
    return count


def isIsomorphic_205(s: str, t: str) -> bool:
    temp = {}
    values = []
    for i in range(len(s)):
        if s[i] not in temp:
            if t[i] in values:
                return False
            temp[s[i]] = t[i]
            values.append(t[i])
        else:
            if temp[s[i]] != t[i]:
                return False
    return True


def reverseList_206(head: ListNode) -> ListNode:
    """
    迭代原链表，新建node查到新链表最前端
    :param head:
    :return:
    """
    if not head:
        return None
    dummy_head = ListNode(-1)
    cur = head
    while cur:
        temp = ListNode(cur.val)
        temp.next = dummy_head.next
        dummy_head.next = temp
        cur = cur.next
    return dummy_head


def reverseList_206_2(head: ListNode) -> ListNode:
    """
    递归实现
    :param head:
    :return:
    """
    if not head or not head.next:
        return head
    # 递归到最后节点
    _next = reverseList_206_2(head.next)
    # head.next是当前节点head的下一个节点，反转就是将head.next的next指向当前节点
    head.next.next = head
    # 断链，防止链表循环，此时head是链表的最后一个结点
    head.next = None
    return _next


def canFinish_207(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    定义：numCourses是结点数，prerequisites：边
    问题: 判断图是否有环

    状态：
         1：当前节点启动的dfs访问过
        -1：其他节点启动dfs访问过
        0： 没有访问过
    方法：对每个节点使用dfs遍历，如果有环，则不能完成返回false

    1.dfs判断有环的条件（终止条件）：
        1.如果visited[node] == 1，说明之前以某个节点开始的dfs，访问过node了，这是第二次访问到该节点，说明有环，返回True。
        2.visited[node] == -1，说明当前节点已经被其他节点开启的dfs访问过，不需要再考虑这个了，返回False。
    2. 设置当前节点为1，
    3. 遍历当前节点node的所有邻接点，判断如果有环，结束返回True。
    4. 当所有邻接节点访问完，没有发现环，则设置当前节点为-1，并且返回False。

    :param numCourses:
    :param prerequisites:
    :return:
    """

    def dfs(graph, node, visited):
        """
        判断是否有环
        思路：
        """
        if visited[node] == 1:
            return True
        if visited[node] == -1:
            return False
        visited[node] = 1
        for i in graph[node]:
            if dfs(graph, i, visited):
                return True
        visited[node] = -1
        return False

    # 构造图(临界表法)
    graph = {}
    for i in range(numCourses):
        graph[i] = []

    for ege in prerequisites:
        head, tail = ege
        graph[head] += [tail]

    # 标志是否访问过, 0,未访问过;1，其他节点访问；-1本节点访问过；
    visited = [0] * numCourses

    # 有向图判断是否有环：深度遍历，如果重复则有环
    for i in range(numCourses):
        if dfs(graph, i, visited):
            return False
    return True


def canFinish_207_2(numCourses: int, prerequisites: List[List[int]]) -> bool:
    if not prerequisites:
        return True

    def _dfs(graph, node, visited, flag):
        """
        回溯算法，判断是否有环
        有环，return False
        """
        # 剪枝，如果已经访问过node，就不需要继续从node开始执行dfs
        if flag[node] == 1:
            return True
        if node not in graph:
            return True
        nodes = graph[node]
        flag[node] = 1
        for node in nodes:
            if node in visited:
                return False
            if not _dfs(graph, node, visited + [node], flag):
                return False
        return True

    # 构造图(临界表法)
    graph = {}
    for ege in prerequisites:
        head, tail = ege
        if head in graph:
            graph[head] += [tail]
        else:
            graph[head] = [tail]
    # 标志是否访问过, 0,未访问;1，以此节点开始执行过dfs。
    flag = [0] * numCourses

    # 有向图判断是否有环：深度遍历，如果重复则有环
    for i in range(numCourses):
        if not _dfs(graph, i, [i], flag):
            return False

    return True


def canFinish_207_3(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    采用拓扑排序判断图是否有环
    拓扑排序算法：
    1. 从DAG中选择一个没有前驱（入度为0）的顶点并输出。，
    2. 从图中删除该顶点和所有以它为起点的有向边。并更新邻接点的入度。
    3. 重复1和2，直到当前节点全部访问完，或者当前图中不存在无前驱的顶点为止（这种情况说明有环）。
    """

    # 构造图(临界表法)
    graph = {}
    for ege in prerequisites:
        head, tail = ege
        if head in graph:
            graph[head] += [tail]
        else:
            graph[head] = [tail]

    # 计算所有节点的入度
    indegree = [0] * numCourses
    for edg in prerequisites:
        indegree[edg[1]] += 1

    # 入度为0的入栈
    stack = []
    for i, v in enumerate(indegree):
        if not v:
            stack.append(i)
    # 记录访问过的节点
    visited = [0] * numCourses
    # 1.
    while stack:
        node = stack.pop()
        numCourses -= 1
        visited[node] = 1
        # 删除到邻接节点的边，同时邻接节点入度-1
        nerbs = graph.get(node)
        if nerbs:
            for n in nerbs:
                indegree[n] -= 1

        # 所有入度为0的重新进栈
        stack = []
        for i, v in enumerate(indegree):
            if not v and not visited[i]:
                stack.append(i)

    return numCourses == 0


class MuliTreeNode(object):
    def __init__(self, val):
        self.val = val
        self.kids = []
        self.is_word = 0


# 208
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = MuliTreeNode('')

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self.head
        for w in word:
            is_in = False
            if cur.kids:
                for kid in cur.kids:
                    if kid.val == w:
                        is_in = True
                        cur = kid
                if not is_in:
                    new = MuliTreeNode(w)
                    cur.kids += [new]
                    cur = new
            else:
                new = MuliTreeNode(w)
                cur.kids = [new]
                cur = new
        cur.is_word = 1

    def _startsWith(self, root, word):
        """
        1. 如果word为空，true
        2. val不为head，val != word[0],返回false
        3. word去掉第一位，递归kids
        4. 最后，如果word还剩字符，返回false
        """
        if not word:
            return True
        if root.val and root.val != word[0]:
            return False
        word = word[1:]
        for kid in root.kids:
            if self._startsWith(kid, word):
                return True
        # 如果最后字符
        if word:
            return False
        else:
            return True

    def _search(self, root, word):
        """
        1. 如果word为空，true
        2. val不为head，val != word[0],返回false
        3. word去掉第一位，递归kids
        4. 最后，如果word还剩字符，返回false
        """
        if not word:
            return False
        if root.val and root.val != word[0]:
            return False
        word = word[1:]
        for kid in root.kids:
            if self._search(kid, word):
                return True

        # 如果最后字符
        if word:
            return False
        elif root.is_word == 1:
            return True
        else:
            return False

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if not word:
            return True
        return self._search(self.head, 's' + word)

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        return self._startsWith(self.head, 's' + prefix)


def minSubArrayLen_209(s: int, nums: List[int]) -> int:
    """
    双指针
    """
    if not nums:
        return 0
    left = 0
    right = 0
    mini_len = len(nums)
    sum_temp = 0
    is_above = False
    while right < len(nums):
        sum_temp += nums[right]
        if sum_temp >= s:
            is_above = True
            mini_len = min(mini_len, right - left + 1)
            # 更新left，如果left-right之间的和>=s,前进left
            while left < right:
                sum_temp -= nums[left]
                left += 1
                if sum_temp < s:
                    break
                mini_len = min(mini_len, right - left + 1)
        right += 1
    if not is_above:
        return 0
    return mini_len


def minSubArrayLen_209_1(s: int, nums: List[int]) -> int:
    if not nums:
        return 0
    L = len(nums)
    left, right, temp_sum = 0, 0, 0
    mini_len = L + 1
    while right < L:
        temp_sum += nums[right]
        while temp_sum >= s:
            mini_len = min(mini_len, right - left + 1)
            temp_sum -= nums[left]
            left += 1
        right += 1

    if mini_len > L:
        return 0
    else:
        return mini_len


def containsNearbyAlmostDuplicate_220(nums: List[int], k: int, t: int) -> bool:
    """
    超时的方法
    如果k>t，另外处理
    """
    i = 0
    L = len(nums)

    while i < L - 1:
        for step in range(1, k + 1):
            if i + step == L - 1:
                end = L - 1
            else:
                end = i + step
            if end == L:
                break
            if math.fabs(nums[i] - nums[end]) <= t:
                return True
        i += 1

    return False


def containsNearbyAlmostDuplicate_220_11(nums: List[int], k: int, t: int) -> bool:
    # 是否有差绝对值为t的两个数
    target = {}
    for index, num in enumerate(nums):
        if (target.get(t + num) is not None and math.fabs(index - target.get(t + num)) <= k) or \
                (target.get(-t + num) is not None and math.fabs(index - target.get(-t + num)) <= k):
            return True
        target[num] = index
    return False


def containsNearbyAlmostDuplicate_220_1(nums: List[int], k: int, t: int) -> bool:
    """
    使用dict（类似two-sum的思路），
    1.对于每个元素，我们想知道之前是否存储了某一个范围（t）内的元素 -> 桶（一个桶之内的任意两个数之差小于等于t）
    2.下标之差小于等于k -> 保证只有k个桶
    :param nums:
    :param k:
    :param t:
    :return:
    """
    if t < 0:
        return False
    # 桶的大小
    sz = t + 1

    def _get_id(num):
        return num // sz

    # 存放桶id和对应的元素
    bucket = {}
    for i in range(len(nums)):
        # 获取桶的id
        id = _get_id(nums[i])
        # 如果id桶中有元素，说明nums[i]在id桶内，返回true
        if id in bucket:
            return True
        # 查看左右相邻桶，如果值在<=t则满足，返回true
        if (id + 1) in bucket and math.fabs(bucket.get(id + 1) - nums[i]) <= t:
            return True

        if (id - 1) in bucket and math.fabs(bucket.get(id - 1) - nums[i]) <= t:
            return True

        # 将该值放入桶中
        bucket[id] = nums[i]
        # 为了满足下标之差<=k的条件。需要始终保持最多有k个桶，k之后的将nums[i-k]的桶移除
        if i >= k:
            remove_id = _get_id(nums[i - k])
            bucket.pop(remove_id)
    return False


def findKthLargest_215(nums: List[int], k: int) -> int:
    """
    1. 构建大根堆
    2. 取k-1词堆顶元素，再取堆顶元素就是
    :param nums:
    :param k:
    :return:
    """

    def siftup(heap, pos):
        """
        ”下沉“操作，
        从pos位置开始，循环递归向下替换上来更大的子节点，直到叶子节点。”下沉“可以理解为把pos位置的值下沉到合适位置。
        """
        endpos = len(heap)
        while pos < endpos:
            lchild = 2 * pos + 1  # 默认左孩子上移
            if lchild >= endpos:  # 如果作孩子是叶子节点，结束
                break
            childpos = lchild
            rchild = 2 * pos + 2  # 如果右孩子更大，则上移右孩子
            if rchild < endpos and heap[childpos] < heap[rchild]:
                childpos = rchild
            if heap[pos] >= heap[childpos]:  # 如果当前节点大于所有孩子节点，结束
                break
            heap[pos], heap[childpos] = heap[childpos], heap[pos]  # 交换
            pos = childpos  # 更新pos，循环

    def build_heap(heap):
        """
        从最后一个非叶子节点开始到堆顶，依次执行”下沉“操作
        """
        n = len(heap)
        for i in reversed(range(n // 2)):
            siftup(heap, i)

    def heappop(heap):
        """
        1.把最后一个位置元素放到堆顶（heap[0]）
        2. 从0位置开始执行”下沉“操作，重新排序
        """
        lastitem = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = lastitem
            siftup(heap, 0)
            return returnitem
        else:
            return lastitem

    # 构建大根堆
    build_heap(nums)
    print(nums)
    # 执行k-1此pop操作
    for _ in range(k - 1):
        heappop(nums)

    return nums[0]


def combinationSum3_216(k: int, n: int) -> List[List[int]]:
    """
    回溯
    1. 结束条件（剪纸）
    2. 判断满足条件记下来
    3. 循环-迭代
    """
    result = []

    def _get_com(sum, level, com, index):
        if level > k or sum > n:
            return
        if level == k and sum != n:
            return
        if sum == n and level == k:
            result.append(com)
            return

        for i in range(index + 1, 10):
            _get_com(sum + i, level + 1, com + [i], i)
        return

    _get_com(0, 0, [], 0)
    return result


import itertools


def combinationSum3_2161(k: int, n: int) -> List[List[int]]:
    return [com for com in itertools.combinations(range(1, 10), k) if sum(com) == n]


def maximalSquare_221(matrix: List[List[str]]) -> int:
    """
    动态规划
    定义：dp[i][j]，以（i,j）为右下角的最大全是”1“正方形边长
    初始化：都为0
    方程： 当当前位置为‘1’，且上左都是1，周边最小+1
          dp[i][j] = min([dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]]) + 1
    """
    if not len(matrix):
        return 0
    H = len(matrix)
    L = len(matrix[0])

    dp = [[int(matrix[i][0])] + [0] * (L - 1) for i in range(1, H)]
    first_col = [int(v) for v in matrix[0]]
    dp.insert(0, first_col)
    if H == 1:
        return max(dp[0])
    max_square = max(dp[0])
    for i in range(1, H):
        for j in range(L):
            if matrix[i][j] == '1':
                dp[i][j] = 1
                # 如果上，左存在且都是'1'，
                if i > 0 and matrix[i - 1][j] == '1' and j > 0 and matrix[i][j - 1] == '1':
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                max_square = max(dp[i][j], max_square)
    print(dp)
    return max_square ** 2


def countNodes_222(root: TreeNode) -> int:
    """
    层次遍历
    :param root:
    :return:
    """
    if not root:
        return 0
    cur = root
    queue = [cur]
    level = 0
    sum = 1
    while queue:
        temp_queue = []
        while queue:
            top = queue.pop(0)
            if top.left and top.right:
                temp_queue.append(top.left)
                temp_queue.append(top.right)
            else:
                # 提前结束
                if top.left:
                    temp_queue.append(top.left)
                sum += len(temp_queue)
                return sum
        level += 1
        sum += 2 ** level
        queue = temp_queue
    return sum


def countNodes_222_1(root: TreeNode) -> int:
    """
    递归计算二叉树节点数量
    :param root:
    :return:
    """
    if not root:
        return 0
    return countNodes_222(root.left) + countNodes_222(root.right) + 1


def computeArea_223(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
    # 计算两个长方形面积
    r1_area = (C - A) * (D - B)
    r2_area = (G - E) * (H - F)
    # 判断是否有交叉（内部重叠也算交叉）,返回area1+area2 - cover
    is_cover = (E < C < G or E < A < G or (A >= E and G >= C) or (E >= A and C >= G)) and (
            F < B < H or F < D < H or (H >= D and B >= F) or (D >= H and F >= B))
    # 判断是否内部, 返回外部矩阵的area
    is_in1 = A > E and C < G and B > F and D < H  # 1在2内部
    is_in2 = E > A and G < C and F > B and H < D  # 2在1内部
    if is_in1:
        return r2_area
    elif is_in2:
        return r1_area
    elif is_cover:
        # 取重叠部分长度 （1.内部，2.交叉）
        if (A >= E and G >= C) or (E >= A and C >= G):
            a = min(C - A, G - E)
        elif C > E > A:
            a = C - E
        else:
            a = G - A
        # 取重叠部分宽度（1.内部，2.交叉）
        if (H >= D and B >= F) or (D >= H and F >= B):
            b = min(D - B, H - F)
        elif B < H < D:
            b = H - B
        else:
            b = D - F
        cover_area = a * b
        return r1_area + r2_area - cover_area
    else:
        return r1_area + r2_area


def computeArea_223_1(A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
    """
    两个矩阵面积之和-重叠部分面积
    :return:
    """

    # 计算重叠部分面积
    def _cover_area():
        # 重叠矩阵左下坐标, 取两个矩阵左下角坐标的最大值
        a_x, a_y = max(E, A), max(F, B)
        # 重叠矩右上坐标，取两个矩阵右上角坐标的最小值
        b_x, b_y = min(C, G), min(D, H)
        # 判断是否重叠
        if a_x <= b_x and a_y <= b_y:
            return (b_x - a_x) * (b_y - a_y)
        return 0

    # 计算两个长方形面积
    r1_area = (C - A) * (D - B)
    r2_area = (G - E) * (H - F)
    return r1_area + r2_area - _cover_area()


def findOrder_210(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # 生成邻接链表（反序的）
    graph = {}
    for i in range(numCourses):
        graph[i] = []

    for edge in prerequisites:
        graph[edge[1]] += [edge[0]]
    # print(graph)
    # 遍历的节点
    nodes = []

    def _dfs(graph, visited, node):
        """
         深度遍历并判断是否有环
         visited三种状态：
         1：当前节点启动的dfs访问过
         -1：其他节点启动dfs访问过
         0： 没有访问过
         """
        if visited[node] == -1:
            return False
        if visited[node] == 1:
            return True
        visited[node] = 1
        for next_node in graph[node]:
            if _dfs(graph, visited, next_node):
                return True
        visited[node] = -1
        nodes.append(node)
        return False

    visited = [0] * numCourses
    for node, edges in graph.items():
        if _dfs(graph, visited, node):
            return []
    if nodes:
        return nodes[::-1]
    else:
        return []


def findOrder_210_1(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    使用拓扑排序
    拓扑排序算法：
    1. 从DAG中选择一个没有前驱（入度为0）的顶点并输出。，
    2. 从图中删除该顶点和所有以它为起点的有向边。并更新邻接点的入度。
    3. 重复1和2，直到当前节点全部访问完，或者当前图中不存在无前驱的顶点为止（这种情况说明有环）。
    """
    # 生成邻接链表（反序的）
    # 构建入度表
    indegree = {}
    graph = {}
    for i in range(numCourses):
        graph[i] = []
        indegree[i] = 0

    for edge in prerequisites:
        graph[edge[1]] += [edge[0]]
        indegree[edge[0]] += 1

    zero_indegree = []
    for k, v in indegree.items():
        if not v:
            zero_indegree.append(k)
    visited_nodes = []
    while zero_indegree:
        # 任取一个零入度的节点
        zero_node = zero_indegree.pop(0)
        visited_nodes.append(zero_node)
        # 取相邻节点
        nei_nodes = graph.get(zero_node)
        # 更新入度表和零入度
        for n in nei_nodes:
            degree = indegree[n] - 1
            indegree[n] = degree
            if not degree:
                zero_indegree.append(n)
    if len(visited_nodes) == numCourses:
        return visited_nodes
    else:
        return []


# 211
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = MuliTreeNode("")

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        word加入前缀树：
        1. 对于word每个字母，从头开始遍历前缀树，如果当前字符在kids中，则继续。否则构造新的节点放在kids中，继续。将最后节点设置is_word标志。
        """
        cur = self.head
        for w in word:
            # 检查是否在当前层的kids中,如果在直接更新cur
            is_in = False
            for kid in cur.kids:
                if kid.val == w:
                    cur = kid
                    is_in = True
                    break
            if is_in:
                continue
            # 不在kids中，加入，并更新cur
            node = MuliTreeNode(w)
            cur.kids.append(node)
            cur = node
        cur.is_word = 1

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        使用dfs查找，
        结束条件：word只剩一个字符，这个字符与当前节点值相同或者是‘.’，并且当前节点有结束标识（is_word==1）
        """

        def _dfs(cur, word):
            # word是最后一个字符，并且是完整标识，
            if not word or not cur:
                return False
            for kid in cur.kids:
                if kid.is_word == 1 and len(word) == 1 and (kid.val == word[0] or word[0] == '.'):
                    return True
                elif word[0] == '.' or kid.val == word[0]:
                    if _dfs(kid, word[1:]):
                        return True
            return False

        return _dfs(self.head, word)


# 225
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q1 = []
        self.q2 = []

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        if not self.q2:
            self.q1.append(x)
        elif not self.q1:
            self.q2.append(x)
        else:
            self.q1.append(x)

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        if not self.q1 and not self.q2:
            return None
        if not self.q2:
            if len(self.q1) == 1:
                return self.q1.pop()
            while len(self.q1) == 1:
                top = self.q1.pop(0)
                self.q2.append(top)
            return self.q1.pop()
        if not self.q1:
            if len(self.q2) == 1:
                return self.q2.pop()
            while len(self.q2) == 1:
                top = self.q2.pop(0)
                self.q1.append(top)
            return self.q2.pop()

    def top(self) -> int:
        """
        Get the top element.
        """
        if not self.q1 and not self.q2:
            return None
        res = None
        if not self.q2:
            while self.q1:
                res = self.q1.pop(0)
                self.q2.append(res)
        if not self.q1:
            while self.q2:
                res = self.q2.pop(0)
                self.q1.append(res)
        return res

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.q1 and not self.q2


def invertTree_226(root: TreeNode) -> TreeNode:
    """
    层次遍历
    :param root:
    :return:
    """
    if not root:
        return
    cur = root
    queue = [cur]
    while queue:
        temp_queue = []
        while queue:
            top = queue.pop(0)

            # 交换左右
            temp = top.left
            top.left = top.right
            top.right = temp

            if top.left:
                temp_queue.append(top.left)
            if top.right:
                temp_queue.append(top.right)

        queue = temp_queue
    return root


def invertTree_226_2(root: TreeNode) -> TreeNode:
    """
    前序遍历实现
    :param root:
    :return:
    """
    if not root:
        return None
    # 保存右子树
    _right_root = root.right
    # 交换左右子树
    root.right = invertTree_226_2(root.left)
    root.left = invertTree_226_2(_right_root)
    return root


def invertTree_226_3(root: TreeNode) -> TreeNode:
    if not root:
        return None
    root.left, root.right = root.right, root.left
    invertTree_226_3(root.left)
    invertTree_226_3(root.right)
    return root


def calculate_227(s: str) -> int:
    """
    方案:
    1. 按照操作符优先级（先‘*’和‘/’，在‘+’，‘-’），每次处理一个操作符+两边的两个数字（num1, op, num2），
    2. 将上面结果替换到原来字符串中
    3. 循环，当没有操作符（以‘-’开头不算）
    注意：
    需要特殊处理
    根据优先级，先处理以‘-’开头的情形

    """
    s = s.replace(' ', '')
    s = s.replace('+0', '')
    s = s.replace('-0', '')

    if not s:
        return 0
    if '*' not in s and '/' not in s and '+' not in s and '-' not in s:
        return int(s)

    def _cal1(num1, oper, num2):
        if oper == '+':
            return int(num1) + int(num2)
        if oper == '-':
            return int(num1) - int(num2)
        if oper == '*':
            return int(num1) * int(num2)
        if oper == '/' and num2 != '0':
            return int(num1) // int(num2)

    # 先执行完‘*’和‘/’
    while len(s):
        if '*' in s or '/' in s:
            index1 = s.find('*')
            index2 = s.find('/')
            if index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
            # 向前
            num1_index = index - 1
            while num1_index >= 0 and '9' >= s[num1_index] >= '0':
                num1_index -= 1
            # 考虑负数的特殊情形
            if num1_index == 0 and s[num1_index] == '-':
                num1_index -= 1
            # 向后
            num2_index = index + 1
            while num2_index < len(s) and '9' >= s[num2_index] >= '0':
                num2_index += 1
            num1 = s[num1_index + 1:index]
            num2 = s[index + 1:num2_index]
            oper = s[index]
            res = _cal1(num1, oper, num2)
            s = s[:num1_index + 1] + str(res) + s[num2_index:]
            continue
        if '+' in s or '-' in s:
            # 如果以‘-’开头
            if s.startswith('-'):
                # 不以‘-’开头
                index1 = s.find('+')
                index2 = s[1:].find('-')
                if index2 == -1 and index1 == -1:
                    return int(s)
                if index1 == -1:
                    index = index2 + 1  # 算上开头的‘-’
                elif index2 == -1:
                    index = index1
                else:
                    index = min(index1, index2 + 1)
                # 向前
                num1_index = index - 1
                while num1_index >= 0 and '9' >= s[num1_index] >= '0':
                    num1_index -= 1
                # 开头的‘-’加上
                num1_index -= 1
                # 向后
                num2_index = index + 1
                while num2_index < len(s) and '9' >= s[num2_index] >= '0':
                    num2_index += 1
                num1 = s[num1_index + 1:index]
                num2 = s[index + 1:num2_index]
                oper = s[index]
                res = _cal1(num1, oper, num2)
                s = s[:num1_index + 1] + str(res) + s[num2_index:]
                continue
            else:
                # 不以‘-’开头
                index1 = s.find('+')
                index2 = s.find('-')
                if index2 == 0 and index1 == -1:
                    return int(s)
                if index1 == -1:
                    index = index2
                elif index2 == -1:
                    index = index1
                else:
                    index = min(index1, index2)
                # 向前
                num1_index = index - 1
                while num1_index >= 0 and '9' >= s[num1_index] >= '0':
                    num1_index -= 1
                # 向后
                num2_index = index + 1
                while num2_index < len(s) and '9' >= s[num2_index] >= '0':
                    num2_index += 1
                num1 = s[num1_index + 1:index]
                num2 = s[index + 1:num2_index]
                oper = s[index]
                res = _cal1(num1, oper, num2)
                s = s[:num1_index + 1] + str(res) + s[num2_index:]
                continue
        break
    return int(s)


def longestCommonPrefix_14(strs: List[str]) -> str:
    if not strs:
        return ''
    i = 0
    zips = zip(*strs)
    for com in zips:
        print(com)
        com = set(com)
        if len(com) == 1:
            i += 1
            continue
        break
    return strs[0][:i]


def reverseList_206_1(head: ListNode) -> ListNode:
    """
    反转链表
    :param head:
    :return:
    """
    if not head:
        return None
    pre, cur = None, head
    while cur:
        # 暂存cur的后继
        temp = cur.next
        # cur指向前驱
        cur.next = pre
        # 更新cur和pre
        pre, cur = cur, temp
    return pre


def summaryRanges_228(nums: List[int]) -> List[str]:
    if not nums:
        return []
    L = len(nums)
    result = []
    first = 0
    second = 0
    while second < L:
        while second < L - 1 and nums[second] + 1 == nums[second + 1]:
            second += 1

        if first == second:
            result.append(str(nums[first]))
        else:
            result.append(str(nums[first]) + '->' + str(nums[second]))
        # second前移一步，
        second += 1
        first = second

    return result


def majorityElement_229(nums: List[int]) -> List[int]:
    """
    借助Counter
    """
    count = Counter(nums)
    result = []
    below = len(nums) // 3
    for k, c in count.items():
        if c > below:
            result.append(k)
    return result


def majorityElement_229_1(nums: List[int]) -> List[int]:
    """
    用摩尔投票法
    首先一点，要选择出现次数大于n/3的数字，最多只能有两个。
    所以问题转化为：1.寻找得票最高的两个数() 2.判断得票数是否大于n/3。
    第一步，使用”摩尔投票法“来选择得票最高的前n个数

    """
    # 计算得票最多的两个候选数字，初始化
    candi1, candi2, c1_count, c2_count = 0, 0, 0, 0
    for num in nums:
        # 更新candi1的票数
        if num == candi1:
            c1_count += 1
            continue
        # 更新candi2的票数
        if num == candi2:
            c2_count += 1
            continue
        # 如果不是candi1和candi2，更换candi1，并更新票数
        if c1_count == 0:
            candi1 = num
            c1_count = 1
            continue
        # 同理更换candi1，并更新票数
        if c2_count == 0:
            candi2 = num
            c2_count = 1
            continue
        # 如果都不满足上面条件，说明出现了新的数字，并且此时最大票数的两个数还是candi1和candi2，因为票数都大于0的
        c1_count -= 1
        c2_count -= 1

    print("candi1:%s, candi2:%s" % (candi1, candi2))

    # 遍历计算两个候选人各自的的票数
    c1_count, c2_count = 0, 0
    for num in nums:
        if num == candi1:
            c1_count += 1
            continue
        if num == candi2:
            c2_count += 1
            continue
    # 选出得票数大于len(nums) // 3的候选人
    result = []
    if c1_count > len(nums) // 3:
        result.append(candi1)
    if c2_count > len(nums) // 3:
        result.append(candi2)
    return result


def kthSmallest_230(root: TreeNode, k: int) -> int:
    """
    中序遍历递归，然后取第k个元素
    """
    result = []

    def _search(root):
        if not root:
            return
        _search(root.left)
        result.append(root.val)
        _search(root.right)

    _search(root)
    return result[k - 1]


def kthSmallest_230_1(root: TreeNode, k: int) -> int:
    """
    中序遍历非递归
    """
    if not root:
        return None
    stack = []
    cur = root
    i = 1
    while stack or cur is not None:
        # 一直往左入栈
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        # 栈顶出栈
        top = stack.pop()
        if i == k:
            return top.val
        i += 1
        # 右孩子入栈
        cur = top.right

    return None


def isPowerOfTwo_231(n: int) -> bool:
    if n <= 0:
        return False
    while n > 1:
        if n % 2 != 0:
            return False
        n = n // 2

    return True


def longestConsecutive_128(nums: List[int]) -> int:
    """
    以每个位置数字为起点，循环讲数字+1然后看是否在nums中，如果在次数+1。转到下一个位置继续，然后取最大次数 (超时)
    优化：1. 比如[100, 4, 200, 1, 3, 2], 在考虑1时，会依次考虑3,2，所有3，2就不需要再考虑了。
        2. 用set先给nums去重
    """
    if not nums:
        return 0
    nums = set(nums)  # 用set去重
    temp = {}  # 定义：{num：以num为起点的最大连续子序列}
    for num in nums:
        if num in temp:
            continue
        temp[num] = 1
        cur = num + 1
        while cur in nums:
            temp[num] += 1
            temp[cur] = 1
            cur += 1
    return max(temp.values())


def longestConsecutive_128_1(nums: List[int]) -> int:
    """
    动态规划
    """
    temp = {}  # 当前元素：以当前元素为起点或者终点的续序列区间长度
    nums = set(nums)
    max_len = 0
    for num in nums:
        if num in temp:
            continue
        # 取以num为结尾的连续序列区间长度
        l_range = temp.get(num - 1, 0)
        # 去以num为开始的连续序列区间长度
        r_range = temp.get(num + 1, 0)
        # 加入num之后，更新区间长度
        cur_range = l_range + r_range + 1
        # 更新最大长度
        if cur_range > max_len:
            max_len = cur_range

        # num为区间[num-lLen, num+rLen]中的值
        # 记下访问过的num，这里num对应以num为起点的子序列长度随便设置一个即可。
        # 因为在找区间的时候只会找到num所在的连续序列的左右端点
        temp[num] = -1
        # 更新左端点为开始的连续序列区间长度
        temp[num - l_range] = cur_range
        # 更新右端点为结尾的连续序列区间长度
        temp[num + r_range] = cur_range
    print(temp)
    return max_len


def isPalindrome_234(head: ListNode) -> bool:
    """
    先把val取出来，然后判断数组是否回文
    """

    def _is_pal(nodes):
        return nodes == nodes[::-1]

    nodes = []
    cur = head
    while cur:
        nodes.append(cur.val)
        cur = cur.next
    return _is_pal(nodes)


def isPalindrome_234_1(head: ListNode) -> bool:
    """
    1.快慢指针找到中间位置
    2. 用栈暂存
    """
    if not head:
        return True
    stack = []
    # 找到中间节点
    first = head
    second = head
    while second and second.next:
        stack.append(first.val)
        first = first.next
        second = second.next.next

    # 双数，单数节点需要后移一步first
    if second is not None:
        first = first.next
    # 　后半部分与前面的值比较
    while first:
        top = stack.pop()
        if first.val != top:
            return False
        first = first.next
    return True


def productExceptSelf_238(nums: List[int]) -> List[int]:
    """
    不让用除法，
    """
    if not nums:
        return []
    L = len(nums)
    result = [1]
    # 计算下矩阵
    temp = 1
    for i in range(L - 1):
        temp *= nums[i]
        result.append(temp)

    # 计算上办矩阵
    temp = 1
    for j in range(L - 1, 0, -1):
        temp *= nums[j]
        result[j - 1] *= temp

    return result


def firstMissingPositive_41(nums: List[int]) -> int:
    """
    用一个数组exits来存连续数，初始化为-1表示未占用.
    遍历nums，把大于等于1并且小于等于数组长度的num在exits数组中对号入座。然后第一个未占用的位置就是结果，如果全部被占用，说明nums刚好是完全连续的。
    解释：为什么需要“小于等于数组长度”？
        如果nums是恰好完全连续的，比如:[1,3,2],这种情况可以完全填充exits数组。
        如果数组：[1,3,2,10]，10大于数组长度，不需要放进exits中。

    """
    if 1 not in nums:
        return 1
    exits = [-1] * len(nums)
    for num in nums:
        # 大于等于1并且小于等于数组长度的记下来
        if 1 <= num <= len(nums):
            exits[num - 1] = num

    res_index = -1
    for index, num in enumerate(exits):
        if num == -1:
            res_index = index + 1
            break
    if res_index == -1:
        return exits[-1] + 1
    else:
        return res_index


def minPathSum_64(grid: List[List[int]]) -> int:
    """
    dp[i,j]：到(i,j)位置最小路径和
    dp[i,j] = min(dp[i-1,j],dp[i,j+1]) + grid[i,j]
    """
    if not grid:
        return 0
    dp = [[0] * len(line) for line in grid]
    # 初始化第一行
    temp = 0
    for i in range(len(grid[0])):
        temp += grid[0][i]
        dp[0][i] = temp
    # 初始化第一列
    temp2 = 0
    for j in range(len(grid)):
        temp2 += grid[j][0]
        dp[j][0] = temp2

    for m in range(1, len(grid)):
        for n in range(1, len(grid[0])):
            dp[m][n] = min(dp[m - 1][n], dp[m][n - 1]) + grid[m][n]

    return dp[-1][-1]


def reverseBetween_92(head: ListNode, m: int, n: int) -> ListNode:
    def _reverse_l(head):
        if not head:
            return head
        cur, pre = head, None
        while cur:
            # 暂存cur的后继
            temp = cur.next
            # cur指向pre
            cur.next = pre
            # 更新cur,pre
            cur, pre = temp, cur
        # head即是反转后的第一个节点
        return pre, head

    if not head:
        return head

    dummy = ListNode(-1)
    dummy.next = head
    cur = head
    i = 1
    middle_after = dummy
    middle = None
    after_after = head
    after = None
    while cur:
        if i == m:
            middle = cur
            # 断开
            middle_after.next = None
        elif i < m:
            middle_after = middle_after.next

        if i == n:
            after = cur.next
            # 断开
            cur.next = None
            break
        i += 1
        cur = cur.next
        after_after = after_after.next

    # 拼接
    reversed_middle, last_node = _reverse_l(middle)
    middle_after.next = reversed_middle
    last_node.next = after
    return dummy.next


def reverseBetween_92_1(head: ListNode, m: int, n: int) -> ListNode:
    if not head:
        return head
    dummy = ListNode(-1)
    dummy.next = head
    # 确定反转区域的前驱节点
    pre = dummy
    for i in range(m - 1):
        pre = pre.next
    cur = pre.next

    # 保持前驱不变
    for j in range(m, n):
        # 暂存后继节点
        next_node = cur.next
        # 当前指向后继的下一个节点（）
        cur.next = next_node.next
        # 后继指向固定前驱下一个（也就是放在反转区域第一个位置）
        next_node.next = pre.next
        # 固定前驱指向后继（保持链接）
        pre.next = next_node

    return dummy.next


def candy_135(ratings: List[int]) -> int:
    """
    贪心算法，
    左规则：从左到右看，如果rating[i]>rating[i-1],i需要比i-1的糖果多一个
    右规则：从右到左看，如果rating[i]>rating[i+1],i需要比i+1的糖果多一个
    “相邻的学生中，评分高的学生必须获得更多的糖果” 等价于，同时满足“左规则”和“右规则”
    """
    if not ratings:
        return 0
    L = len(ratings)
    # 左 -> 右
    left = [1] * L
    for i in range(1, L):
        if ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1
    print(left)
    # 右 -> 左
    right = [1] * L
    for j in range(L - 2, -1, -1):
        if ratings[j] > ratings[j + 1]:
            right[j] = right[j + 1] + 1
    print(right)

    sum = 0
    for i in range(L):
        sum += max(left[i], right[i])

    return sum


def preorderTraversal(root: TreeNode) -> List[int]:
    res = []

    def _preorder(root):
        # 递归
        if not root:
            return
        res.append(root.val)
        _preorder(root.left)
        _preorder(root.right)

    def _preorder2(root):
        # 非递归
        if not root:
            return []
        res = []
        stack = [root]
        cur = root
        while stack:
            res.append(cur.val)
            # 一直向左进栈
            while cur.left:
                res.append(cur.left.val)
                stack.append(cur.left)
                cur = cur.left
            # 出栈,如果有右节点，右节点入栈，结束出栈
            while stack:
                cur = stack.pop()
                if cur.right:
                    stack.append(cur.right)
                    cur = cur.right
                    break
        return res

    def _preorder3(root):
        # 非递归-简化版本
        if not root:
            return []
        res = []
        cur = root
        stack = []
        while stack or cur:
            # 一直往左进栈
            while cur:
                stack.append(cur)
                cur = cur.left
            # 出栈
            cur = stack.pop()
            res += [cur.val]
            cur = cur.right
        return res

    def _preorder4(root):
        if not root:
            return []
        res = []
        stack = [root]
        while stack:
            top = stack.pop()
            res.append(top.val)
            if top.right:
                stack.append(top.right)
            if top.left:
                stack.append(top.left)
        return res

    res = _preorder4(root)
    return res


def shortestSubarray_862(A: List[int], K: int) -> int:
    if not A:
        return -1
    i = 0
    max_len = len(A)
    latest_right = 0
    has_ans = False
    while i < len(A):
        j = max(i + 1, latest_right)
        temp_sum = sum(A[i:j])
        if temp_sum >= K:
            latest_right = j
            max_len = min(max_len, 1)
            has_ans = True
        else:
            while j < len(A):
                temp_sum += A[j]
                if temp_sum >= K:
                    latest_right = j
                    max_len = min(max_len, j - i + 1)
                    has_ans = True
                    break
                j += 1
        i += 1
    if has_ans:
        return max_len
    else:
        return -1


def nextGreaterElement_498(nums1: List[int], nums2: List[int]) -> List[int]:
    def _sorted_stack(nums2):
        stack = []  # 记录原数组的下标
        next_dict = {}
        for index, num in enumerate(nums2):
            next_dict[num] = -1
            while stack and nums2[stack[-1]] < num:
                next_dict[nums2[stack[-1]]] = num
                stack.pop()
            stack.append(index)
        return next_dict

    next_dict_res = _sorted_stack(nums2)

    res = []
    for n in nums1:
        res.append(next_dict_res[n])

    return res


def _method(nums):
    result = [-1] * len(nums)
    mono_stack = []  # 存放下标
    for index, num in enumerate(nums):
        while mono_stack and nums[mono_stack[-1]] < num:
            result[mono_stack[-1]] = index - mono_stack[-1]
            mono_stack.pop()
        mono_stack.append(index)
    return result


def searchMatrix_240(matrix, target):
    """
    与”左下角“比较(与”右上角“比较思路一样)
    ”左下角“是所在行的最小值，所在列的最大值
    1. ”左下角“ = target,返回True
    2. ”左下角“ < target,说明肯定不在这一行中，去掉这一行，再剩下中找
    3. ”左下角“ > target,说明肯定不在这一列中，去掉这一列，在剩下中找
    """
    if not matrix:
        return -1

    # 左下角坐标（i,j）
    i, j = 0, len(matrix) - 1
    while i < len(matrix[0]) and j >= 0:
        print(matrix[j][i])
        if matrix[j][i] == target:
            return True
        elif matrix[j][i] > target:
            j -= 1
        else:
            i += 1
    return False


if __name__ == '__main__':
    tree1 = BTree()
    tree1.create_btree([1, 2, None, 3])
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
    m = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    res = searchMatrix_240(m, 7.8)
    print(res)
