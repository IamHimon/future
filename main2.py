import collections
import math
import operator
import re
import sys

from collections import defaultdict, deque
from my_heapq import heappush, heappop, heapify
from typing import List

from TreeNode import BTree
from base_classes import ListNode, Node, TreeNode


def isValidBST_98(root: TreeNode) -> bool:
    """
    递归中序遍历，然后判断是否升序
    :param root:
    :return:
    """

    def _is_sort(nums):
        return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))

    def _get_values(root, res):
        if root is None:
            return
        _get_values(root.left, res)
        res += [root.value]
        _get_values(root.right, res)

    res = []
    _get_values(root, res)
    return _is_sort(res)


def isValidBST_98_2(root: TreeNode) -> bool:
    """
    stack实现中序遍历，判断
    :param root:
    :return:
    """
    stack = []
    res = []
    if root is None:
        return True
    while root is not None or stack:
        while root is not None:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if res and root.value <= max(res):
            return False
        else:
            res += [root.value]
        root = root.right
    return True


def level_order_102(root: TreeNode):
    """
    层次遍历
    :param root:
    :return:
    """
    if root is None:
        return []
    res = []
    queue = [root]
    while queue:
        temp = []
        temp_queue = []
        while queue:
            cur = queue.pop(0)
            if cur.left:
                temp_queue.append(cur.left)
            if cur.right:
                temp_queue.append(cur.right)
            temp.append(cur.val)
        res.append(temp)
        queue = temp_queue
    return res


def zigzagLevelOrder_103(root: TreeNode):
    if root is None:
        return []
    res = []
    queue = [root]
    reverse = False
    while queue:
        temp = []
        temp_queue = []
        while queue:
            if reverse:
                cur = queue.pop(0)
                if cur.left:
                    temp_queue.append(cur.left)
                if cur.right:
                    temp_queue.append(cur.right)
                temp.insert(0, cur.val)
            else:
                cur = queue.pop(0)
                if cur.left:
                    temp_queue.append(cur.left)
                if cur.right:
                    temp_queue.append(cur.right)
                temp.append(cur.val)
        res.append(temp)
        queue = temp_queue
        reverse = not reverse
    return res


def buildTree_105(preorder: List[int], inorder: List[int]) -> TreeNode:
    """
    105,先序第一个是根结点
    :param preorder:
    :param inorder:
    :return:
    """
    if len(preorder) == 0 or len(inorder) == 0:
        return None

    def _build(root, preorder, inorder):
        if preorder and inorder and root:
            # 切分中序
            root_index = inorder.index(root.value)
            left_inorder = inorder[:root_index]
            right_inorder = inorder[root_index + 1:]

            # 切分前序
            left_preorder = preorder[1:root_index + 1]
            right_preorder = preorder[root_index + 1:]

            if left_preorder:
                root.left = TreeNode(left_preorder[0])
            if right_preorder:
                root.right = TreeNode(right_preorder[0])

            _build(root.left, left_preorder, left_inorder)
            _build(root.right, right_preorder, right_inorder)

    root = TreeNode(preorder[0])
    _build(root, preorder, inorder)
    return root


def buildTree_105_1(preorder: List[int], inorder: List[int]) -> TreeNode:
    """
    精简做法
    :param preorder:
    :param inorder:
    :return:
    """
    if not preorder or not inorder:
        return
    root = TreeNode(preorder[0])
    root_index = inorder.index(preorder[0])
    root.left = buildTree_105_1(preorder[1:root_index + 1], inorder[:root_index])
    root.right = buildTree_105_1(preorder[root_index + 1:], inorder[root_index + 1:])
    return root


def buildTree_106(inorder: List[int], postorder: List[int]) -> TreeNode:
    """
    106,后序最后一个是根结点
    :param inorder:
    :param postorder:
    :return:
    """
    if not inorder or not postorder:
        return
    root = TreeNode(postorder[-1])
    root_index = inorder.index(postorder[-1])
    root.left = buildTree_106(inorder[:root_index], postorder[:root_index])
    root.right = buildTree_106(inorder[root_index + 1:], postorder[root_index:-1])
    return root


def sortedListToBST_109(head: ListNode) -> TreeNode:
    """
    109, 又二分法的思路构建BST，它肯定就是height balance的。
    :param head:
    :return:
    """
    if head is None:
        return
    if head.next is None:
        return TreeNode(head.val)

    # 快慢指针方法来获取当前列表中间结点
    cur = head
    slow = head.next
    fast = head.next.next
    while fast and fast.next:
        cur = cur.next
        slow = slow.next
        fast = fast.next.next

    # cur用于从中间断点
    cur.next = None
    root = TreeNode(slow.val)
    # 递归构造BST
    root.left = sortedListToBST_109(head)
    root.right = sortedListToBST_109(slow.next)
    return root


def hasPathSum_112(root: TreeNode, target: int) -> bool:
    """
    112,
    """
    if root is None:
        return False

    result = []
    flag = False

    def _dfs(root, sum):
        if sum == target and root.left is None and root.right is None:
            result.append(1)
            flag = True
            return
        if root.left:
            _dfs(root.left, sum + root.left.val)
        if root.right:
            _dfs(root.right, sum + root.right.val)
        return

    _dfs(root, root.val)
    print(flag)
    return flag


def pathSum_113(root: TreeNode, target: int) -> List[List[int]]:
    """
    113,
    """
    if root is None:
        return []

    result = []

    def _dfs(root, values, sum):
        if sum == 0 and root.left is None and root.right is None:
            result.append(values)
            return
        if root.left:
            _dfs(root.left, values + [root.left.val], sum - root.left.val)
        if root.right:
            _dfs(root.right, values + [root.right.val], sum - root.right.val)
        return

    _dfs(root, [root.val], target - root.val)
    return result


def flatten_114(root: TreeNode) -> None:
    """
    Do not return anything, modify root in-place instead.
    1. 定位当前结点左子树最右结点
    2. 把右子树接到左子树最右边
    3. 左子树接到当前结点左边
    """
    while root:
        left = root.left
        cur = left
        if cur:
            # 找到左子树最右结点
            while cur.right or cur.left:
                if cur.right:
                    cur = cur.right
                    continue
                if cur.left:
                    cur = cur.left
            # 把右子树接到左子树最右边
            cur.right = root.right

            # 左子树接到当前结点左边
            root.right = left
            root.left = None
        root = root.right


def connect_116(root: Node) -> Node:
    """
    116, 层次遍历,借助队列
    :param root:
    :return:
    """
    if root is None:
        return root
    queue = [root]
    while queue:
        temp_queue = []
        pre_node = queue.pop(0)
        if pre_node.left:
            temp_queue.append(pre_node.left)
        if pre_node.right:
            temp_queue.append(pre_node.right)
        while queue:
            cur = queue.pop(0)
            pre_node.next = cur
            pre_node = cur
            if cur.left:
                temp_queue.append(cur.left)
            if cur.right:
                temp_queue.append(cur.right)
        queue = temp_queue
    return root


def minimumTotal_120(triangle: List[List[int]]) -> int:
    """
    120,动态规划，自下向上。动态规划不一定是从头到尾的。
    """
    if not triangle:
        return 0
    high = len(triangle)
    if high == 1:
        return triangle[0][0]
    for level in range(high - 2, -1, -1):
        for j in range(len(triangle[level])):
            triangle[level][j] = min(triangle[level + 1][j], triangle[level + 1][j + 1]) + triangle[level][j]
    return triangle[0][0]


def minimumTotal_120_2(triangle: List[List[int]]) -> int:
    """
    120,回朔，超时了。
    """
    if not triangle:
        return 0
    max_value = triangle[0][0]
    max_length = len(triangle)
    result = [sys.maxsize]

    def _search(level, max_value, index):
        if level == max_length - 1:
            if max_value < result[0]:
                result[0] = max_value
            return
        _search(level + 1, max_value + triangle[level + 1][index], index)
        _search(level + 1, max_value + triangle[level + 1][index + 1], index + 1)

    _search(0, max_value, 0)

    return result[0]


def ladderLength_127(beginWord: str, endWord: str, wordList: List[str]) -> int:
    """
    这个问题抽象成在无向环图中寻找最短路径的问题。
    关键是如何构造图以及如何定义每个结点的相邻结点。
    1. 通过beginWord，endWord以及wordList构造一个无向环图。相邻结点的定义是只相差一个字符的word。
    2. 某个结点的邻居结点有： 每个位置变换字符后得到的同时在wordList中的所有word。
    3. 使用defaultdict，是当key不存在，会自动创建key-value。
    4. 借助defaultdict来记录已经访问过的结点
    5. 借用queue实现宽度优先搜索
    6.
    :param beginWord:
    :param endWord:
    :param wordList:
    :return:
    """
    # 建立通用list
    size, neighbours_dict = len(beginWord), defaultdict(list)
    # 构造结点每个变换位置word与所有邻居结点的dict，比如："h*t":["hot"],表示，如果是1位置变换，wordList中对应的结点有["hot"]
    for w in wordList:
        for _ in range(size):
            neighbours_dict[w[:_] + "*" + w[_ + 1:]].append(w)

    # BFS
    queue = deque()  # 队列实现宽度优先搜索
    queue.append((beginWord, 1))  # 因为在BFS中，queue中通常会同时混合多层的node，这就无法区分层了，要区分层就要queue中直接加入当前node所属层数。
    visited_dict = defaultdict(bool)  # bool 的默认值是false，因此所有不在list里的是false
    visited_dict[beginWord] = True
    while queue:
        cur_word, level = queue.popleft()  # queue头出来一个
        for i in range(size):  # 找邻居，每个位置变换
            key = cur_word[:i] + "*" + cur_word[i + 1:]  # 每个位置变换后的邻居结点key
            for neighbour in neighbours_dict[key]:
                if neighbour == endWord:
                    return level + 1  # 如果找到endWord，直接返回
                if not visited_dict[neighbour]:
                    visited_dict[neighbour] = True
                    queue.append((neighbour, level + 1))  # 符合条件（neighbour + unmarked)的进去
    return 0


def sumNumbers_129(root: TreeNode) -> int:
    """
    129，回朔算法
    :param root:
    :return:
    """
    max_value = [0]
    if not root:
        return 0

    def _get_path(root, num):
        if not root.left and not root.right:
            max_value[0] += int(num)
            return
        if root.left:
            _get_path(root.left, num + str(root.left.val))
        if root.right:
            _get_path(root.right, num + str(root.right.val))

    _get_path(root, str(root.val))

    return max_value[0]


def solve_130(board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    if len(board) <= 2:
        return

    flag = [[0] * len(col) for col in board]

    hight = len(board)
    broad = len(board[0])

    for k in range(1, hight - 1):
        for p in range(1, broad - 1):
            if board[k][p] == 'O' and flag[k][p] == 0:
                # 上下左右都是'X',直接变换
                if board[k - 1][p] == 'X' and board[k + 1][p] == 'X' and \
                        board[k][p - 1] == 'X' and board[k][p + 1] == 'X':
                    board[k][p] = 'X'
                else:
                    # BFS
                    queue = deque()
                    queue.append((k, p))
                    flag[k][p] = 1
                    visited_nodes = [(k, p)]
                    has_invalid = False
                    while queue:
                        i, j = queue.popleft()
                        # 遍历所有邻居
                        # 上
                        if i - 1 >= 0 and board[i - 1][j] == 'O' and flag[i - 1][j] == 0:
                            flag[i - 1][j] = 1
                            queue.append((i - 1, j))
                            visited_nodes.append((i - 1, j))
                            if i - 1 == 0:
                                has_invalid = True

                        # 下
                        if i + 1 <= hight - 1 and board[i + 1][j] == 'O' and flag[i + 1][j] == 0:
                            flag[i + 1][j] = 1
                            queue.append((i + 1, j))
                            visited_nodes.append((i + 1, j))
                            if i + 1 == hight - 1:
                                has_invalid = True
                        # 左
                        if j - 1 >= 0 and board[i][j - 1] == 'O' and flag[i][j - 1] == 0:
                            flag[i][j - 1] = 1
                            queue.append((i, j - 1))
                            visited_nodes.append((i, j - 1))
                            if j - 1 == 0:
                                has_invalid = True
                        # 右
                        if j + 1 <= broad - 1 and board[i][j + 1] == 'O' and flag[i][j + 1] == 0:
                            flag[i][j + 1] = 1
                            queue.append((i, j + 1))
                            visited_nodes.append((i, j + 1))
                            if j + 1 == broad - 1:
                                has_invalid = True
                    if not has_invalid:
                        # 所有都是有效的批量变换
                        for I, J in visited_nodes:
                            board[I][J] = 'X'
                        k = visited_nodes[-1][0]
                        p = visited_nodes[-1][1]


def plusOne_66(digits: List[int]) -> List[int]:
    """
    1. 逐位相加，
    2. 处理进位,
        末位不进位
        末位进位
    """
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        temp_sum = digits[i] + carry
        carry = temp_sum // 10
        mod = temp_sum % 10
        digits[i] = mod
        if carry and i == 0:
            digits.insert(0, 1)
    return digits


def climbStairs_70(n: int) -> int:
    """
    状态定义：dp[i]表示爬到第i个台阶的方案数
    更新方程：dp[i] = dp[i-2] + dp[i-1],因为只能爬1级或者2级
    :param self:
    :param n:
    :return:
    """
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 2] + dp[i - 1]
    return dp[-1]


def minDepth_111_1(root: TreeNode) -> int:
    """
    递归取所有叶子结点的深度，然后取最小值
    """
    if not root:
        return 0
    all_depth = []

    def _get_depth(root, depth):
        if not root.left and not root.right:
            all_depth.append(depth)
        if root.left:
            _get_depth(root.left, depth + 1)
        if root.right:
            _get_depth(root.right, depth + 1)

    _get_depth(root, 1)
    return min(all_depth)


def minDepth_111_2(root: TreeNode) -> int:
    if not root:
        return 0
    # 计算左子树
    left_dep = minDepth_111_2(root.left)
    # 计算右子树
    right_dep = minDepth_111_2(root.right)
    # 如果有一个为0，+1，否则最小值+1
    if left_dep and right_dep:
        return min(left_dep, right_dep) + 1
    else:
        return left_dep + right_dep + 1


def partition_131(s: str) -> List[List[str]]:
    """
    回朔算法

    回朔三要素：
    1. 有效结果
        到最后一个字符，将当前com加入到result中
    2. 回溯范围及答案更新
        遍历当前位置后的所有子串
    3. 剪枝条件
        是不是回文子串
    """
    result = []

    def _is_pal(s):
        # 判断是不是回文子串
        is_pal = True
        for i in range(len(s) // 2):
            if s[i] != s[len(s) - 1 - i]:
                is_pal = False
                break
        return is_pal

    def _get_com(s, start, com):
        if start >= len(s):
            result.append(com)
            return
        for i in range(start + 1, len(s) + 1):
            temp = s[start:i]
            if _is_pal(temp):
                _get_com(s, i, com + [temp])

    _get_com(s, 0, [])
    return result


def singleNumber_136(nums: List[int]) -> int:
    def _method1():
        """
        借助dict,
        :return:
        """
        temp = {}
        for num in nums:
            if num in temp:
                temp.pop(num)
            else:
                temp[num] = 0
        for k, v in temp.items():
            return k

    def _method2():
        """
        异或运算符，
        1. 0和任何数异或的结果都是这个数本身
        2. 相同的数异或的结果为0
        :return:
        """
        res = 0
        for num in nums:
            res = res ^ num
        return res

    return _method2()


def singleNumber_137(nums: List[int]) -> int:
    def _m1():
        temp = {}
        for num in nums:
            if num in temp:
                temp[num] += 1
            else:
                temp[num] = 1
        for k, v in temp.items():
            if v == 1:
                return k

    return _m1()


def wordBreak_139_1(s: str, wordDict: List[str]) -> bool:
    """
    回溯，超时。
    原因分析：回溯是找到所有结果，遍历所有组合，这样一些情况会比较复杂，而这道题是只需要找到一个结果就行。

    :param s:
    :param wordDict:
    :return:
    """
    result = []
    S = "*" * len(s)
    wordDict = [w for w in wordDict if w in s]

    def _deal(s: str, wordDict):
        if s == S:
            result.append(1)
            return True
        for char in wordDict:
            if char in s:
                _deal(s.replace(char, '*' * len(char)), wordDict)
        return False

    _deal(s, wordDict)
    return len(result) != 0


def wordBreak_139_2(s: str, wordDict: List[str]) -> bool:
    """
    动态规划
    定义：
        dp[i]，表示前i个字符可以正确用wordDict拆分
    初始化：
        dp[0],表示空字符串，是true
    方程：
        dp[i] = dp[j] and check(s[j:i-1])   # 构建i与i-1的方程关系
        check(s[j:i-1])表示判断s[j:i-1]是否在字典中
        j是遍历i-1之前所有的字串

    """

    def _m1(dp, max_len):
        # 反向的
        for i in range(1, len(s) + 1):
            if i - max_len < 0:
                end = -1
            else:
                end = i - max_len - 1
            for j in range(i - 1, end, -1):
                for j in range(i - 1, end, -1):
                    dp[i] = dp[j] and s[j:i] in wordDict
                    if dp[i]:
                        break  # 因为此时不需要再用 j 去把长度 i 的这个子串分成两部分考察了,
                        # 长度 i 的这个子串已经可以 break 成单词表的单词了，j 没必要继续扫描

    def _m2(dp):
        # 正向的,想象成爬楼梯问题
        #
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True

    max_len = 0
    for word in wordDict:
        max_len = max(max_len, len(word))

    dp = [False] * (len(s) + 1)
    dp[0] = True
    _m1(dp, max_len)
    return dp[-1]


def wordBreak_140_1(s: str, wordDict: List[str]) -> List[str]:
    """
    回溯，超时
    :param s:
    :param wordDict:
    :return:
    """
    S = "*" * len(s)
    wordDict = [w for w in wordDict if w in s]
    result = []

    def _deal(s: str, split_s):
        if s == S:
            result.append(' '.join(split_s))
            return True
        for char in wordDict:
            if char in s:
                start_index = s.index(char)
                if s[:start_index] == '*' * start_index:
                    _deal(s.replace(char, '*' * len(char), 1), split_s + [char])
        return False

    _deal(s, [])
    return result


def reorderList_143(head: ListNode) -> None:
    """
    借助栈
    :param head:
    :return:
    """
    if not head:
        return
    stack = []
    cur = head
    while cur:
        stack.append(cur)
        cur = cur.next
    new_head = ListNode(-1)
    new_cur = new_head
    left = 0
    right = len(stack) - 1
    while right >= left:
        if left == right:
            l_node = stack[left]
            left += 1
            new_cur.next = l_node
            new_cur = new_cur.next
        else:
            l_node = stack[left]
            r_node = stack[right]
            left += 1
            right -= 1
            new_cur.next = l_node
            new_cur = new_cur.next
            new_cur.next = r_node
            new_cur = new_cur.next

    new_cur.next = None
    return new_head.next


def isPalindrome_125_1(s: str) -> bool:
    """
    判断是否是回文
    1. 先把非字符去掉，大小写统一
    2. 使用二分法比较new_s
    :param s:
    :return:
    """
    if not s:
        return True
    new_s = ''
    for c in s:
        if c.isalnum():
            new_s += c.lower()
    # 判断是不是回文子串
    is_pal = True
    for i in range(len(new_s) // 2):
        if new_s[i] != new_s[len(new_s) - 1 - i]:
            is_pal = False
            break
    return is_pal


def isPalindrome_125_2(s):
    if not s:
        return True
    start = 0
    end = len(s) - 1
    s = s.lower()
    while start <= end:
        start_is = s[start].isalnum()
        end_is = s[end].isalnum()
        if start_is and end_is:
            if s[start] != s[end]:
                return False
            else:
                start += 1
                end -= 1
        elif not start_is:
            start += 1
        elif not end_is:
            end -= 1
    return True


def isPalindrome_125_3(s: str):
    s = re.sub(r'[^a-z0-9]', '', s.lower().strip())
    return s == s[::-1]


def isInterleave_97(s1: str, s2: str, s3: str) -> bool:
    """
    错误的动规
    没有fc，
    'aa', 'ab', 'aaba'
    :param s1:
    :param s2:
    :param s3:
    :return:
    """

    def _get_over_part(s_full, s_part):
        over = 0
        if not s_part or not s_full:
            return over
        if not s_full.startswith(s_part):
            return over
        return len(s_part)

    l1 = len(s1)
    l2 = len(s2)
    s3 = '-' + s3
    l3 = len(s3)
    max_l = max(l1, l2)
    if l3 != (l1 + l2 + 1):
        return False
    dp = [False] * l3
    dp[0] = True
    i = 1
    while i < l3:
        j = min(i + max_l, l3)
        while j >= i + 1:
            s_part = s3[i:j]
            max_in_s1 = _get_over_part(s1, s_part)  # 从开头重复的位数
            max_in_s2 = _get_over_part(s2, s_part)
            if max_in_s1 > 0 or max_in_s2 > 0:
                dp[j - 1] = dp[i - 1]
                i = j - 1
                if max_in_s1 > max_in_s2:
                    s1 = s1[max_in_s1:]
                    break
                else:
                    s2 = s2[max_in_s2:]
                    break
            j -= 1
        i += 1
    return dp[-1]


def generate_118(numRows: int) -> List[List[int]]:
    result = []
    if not numRows:
        return result
    result.append([1])
    for i in range(1, numRows):
        temp = [1]
        for j in range(1, i):
            temp.append(result[i - 1][j - 1] + result[i - 1][j])
        temp.append(1)
        result.append(temp)
    return result


def getRow_119(rowIndex: int) -> List[int]:
    if rowIndex == 0:
        return [1]
    last_row = [1]
    for i in range(1, rowIndex + 1):
        temp = [1]
        for j in range(1, i):
            temp.append(last_row[j - 1] + last_row[j])
        temp.append(1)
        last_row = temp
    return last_row


class LRUCache:

    def __init__(self, capacity: int):
        self.CAP = capacity
        self.value_cache = []  # 放value
        self.map = {}  # 放key-value下标
        self.first_map = ()

    def get(self, key: int) -> int:
        # key再map中，给结果。并且，把key对应value放在第一位（map中改变）
        if key in self.map:
            res = self.value_cache[self.map.get(key)]
            # 更新
            self._judge(key)
            return res
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        # 还有空间，key不再cache中，直接放进去
        if len(self.value_cache) < self.CAP:
            if key not in self.map:
                self.value_cache.append(value)
                self.map[key] = len(self.value_cache) - 1
                self.first_map = (key, len(self.value_cache) - 1)
            else:
                self.value_cache[self.map.get(key)] = value
        elif len(self.value_cache) == self.CAP:
            # 没空间了，弹出最开始一个,map也弹出对应的
            # 弹出key对应value为0的
            if key not in self.map:
                zero_index = -1
                zeor_key = -1
                for k, i in self.map.items():
                    if i != 0:
                        self.map[k] -= 1
                    else:
                        zero_index = i
                        zeor_key = k
                self.map.pop(zeor_key)
                self.value_cache.pop(zero_index)
                # 放新的k-v
                self.value_cache.append(value)
                self.map[key] = self.CAP - 1
                self.first_map = (key, self.CAP - 1)
            else:
                old_value_index = self.map.get(key)
                self.value_cache[old_value_index] = value
                # 调整下标
                self._judge(key)

    def _judge(self, key):
        """
        当前与最早使用的一个替换，map替换，value_cache替换
        todo:调整下标，要全部调整，把key放在第一位，其他都递减
        :param cur_index: 当前下标
        :param earlies_index:
        :return:
        """
        key_index = self.map.get(key)
        first_index = self.first_map[1]
        first_key = self.first_map[0]
        self.map[key] = first_index
        self.map[first_key] = key_index
        for k, v in self.map.items():
            if k != key:
                self.map[k] -= 1
        # cache中更新
        self.value_cache[key_index], self.value_cache[first_index] = self.value_cache[first_index], \
                                                                     self.value_cache[key_index]
        self.first_map = (key, first_index)


class LRUCache1(collections.OrderedDict):
    def __init__(self, capacity: int):
        self.cap = capacity
        self.key_value = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.key_value:
            self.key_value.move_to_end(key)
            return self.key_value.get(key)
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.key_value:
            self.key_value.move_to_end(key)
        self.key_value[key] = value
        if len(self.key_value) > self.cap:
            self.key_value.popitem(last=False)


def insertionSortList_147(head: ListNode) -> ListNode:
    """
    链表排序，
    另外新建一个新链表，在原串每个位置插入到新串的合适位置中。
    :param head:
    :return:
    """
    if not head:
        return
    new_head = ListNode(-1)
    cur = head
    while cur:
        # 按顺序放进new_head中
        cur_temp = ListNode(cur.val)
        sub_1 = new_head
        sub_2 = new_head.next
        while sub_2:
            if sub_2.val >= cur.val:
                break
            sub_1 = sub_1.next
            sub_2 = sub_2.next
        sub_1.next = cur_temp
        cur_temp.next = sub_2
        # 在原串上前进一步
        cur = cur.next
    return new_head.next


def sortList_148(head: ListNode) -> ListNode:
    """
    归并排序实现，
    :param head:
    :return:
    """

    def _div(head):
        if not head or not head.next:
            return head
        left_half, right_half = _split(head)

        # 递归分割
        # 注意：递归，一定要对两个分支重新赋值，要不然就断链了
        left_half = _div(left_half)
        right_half = _div(right_half)
        # 合并
        return _merge(left_half, right_half)

    def _split(head):
        # 将链表从中间分开
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        right_half = slow.next
        slow.next = None
        left_half = head
        return left_half, right_half

    def _merge(left, right):
        # 按val顺序合并两个链表
        # 使用dummy做新的链表
        dummy = ListNode(0)
        l = left
        r = right
        cur = dummy

        while l and r:
            if l.val <= r.val:
                cur.next = l
                l = l.next
            else:
                cur.next = r
                r = r.next
            cur = cur.next
        if l:
            cur.next = l
        if r:
            cur.next = r
        return dummy.next

    return _div(head)


def maxProduct_152(nums: List[int]) -> int:
    """
    贪婪算法
    :param nums:
    :return:
    """
    if not nums:
        return 0
    min_prod = nums[0]
    max_prod = nums[0]
    _max = nums[0]
    _min = nums[0]
    for num in nums[1:]:
        temp1 = min_prod * num
        temp2 = max_prod * num
        min_prod = min([temp1, temp2, num])
        max_prod = max([temp1, temp2, num])
        _max = max(max_prod, _max)
        _min = min(min_prod, _min)

    return _max


def maxProduct_152_1(nums: List[int]) -> int:
    # 暴力解法,超时
    if not nums:
        return 0
    L = len(nums)
    max = nums[0]
    # 处理1和-1

    # 窗口
    for win in range(1, len(nums) + 1):
        for i in range(L - win + 1):
            temp = 1
            for n in nums[i:i + win]:
                temp *= n
            if temp > max:
                max = temp
    return max


def maxProduct_152_2(nums: List[int]) -> int:
    """
    动态规划
    定义：
    dp[i] = (min,max)
    dp[i][0] = min(dp[i-1][0] * num, dp[i-1][1]*num, num[i])
    dp[i][1] = max(dp[i-1][0] * num, dp[i-1][1]*num, num[i])
    :param nums:
    :return:
    """
    if not nums:
        return 0
    L = len(nums)
    _min = [0] * L
    _max = [0] * L
    _min[0] = nums[0]
    _max[0] = nums[0]
    for i in range(1, L):
        temp1 = _min[i - 1] * nums[i]
        temp2 = _max[i - 1] * nums[i]
        __min = min(temp1, temp2, nums[i])
        __max = max(temp1, temp2, nums[i])
        _max[i] = __max
        _min[i] = __min
    return max(_max)


def findMin_153(nums: List[int]) -> int:
    res = nums[0]
    i = 0
    while i < len(nums) - 1:
        if nums[i] > nums[i + 1]:
            return nums[i + 1]
        i += 1
    return res


def findMin_153_1(nums: List[int]) -> int:
    """
    二分查找,根据一条发现：正常排序的总是最左边数字小于最右边数字，

    :param nums:
    :return:
    """
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        # 当mid > right,则分界点只可能出现再[mid+1,right]中
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1  # 相等时右移，不会对结果造成影响，安全的方法
    return nums[left]


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.sorted_value = []

    def push(self, x: int) -> None:
        self._insert_sort(x)
        self.stack.append(x)

    def _insert_sort(self, x):
        self.sorted_value.append(x)
        if len(self.sorted_value) > 1:
            i = len(self.sorted_value) - 2
            while self.sorted_value[i] > x and i >= 0:
                self.sorted_value[i], self.sorted_value[i + 1] = self.sorted_value[i + 1], self.sorted_value[i]
                i -= 1

    def pop(self) -> None:
        if self.stack and self.sorted_value:
            self.sorted_value.remove(self.stack[-1])
            self.stack = self.stack[:-1]

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        else:
            return None

    def getMin(self) -> int:
        return self.sorted_value[0]


class MinStack1:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.sorted_value = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        heappush(self.sorted_value, x)

    def pop(self) -> None:
        if self.stack and self.sorted_value:
            self.sorted_value.remove(self.stack[-1])
            heapify(self.sorted_value)
            self.stack = self.stack[:-1]

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]
        else:
            return None

    def getMin(self) -> int:
        return self.sorted_value[0]


def getIntersectionNode_160(headA: ListNode, headB: ListNode) -> ListNode:
    """
    求两个链表的公共链表部分
    :param headA:
    :param headB:
    :return:
    """
    if not headA or not headB:
        return None
    a = headA
    while a:
        b = headB
        temp = a
        while b:
            if a.val == b.val:
                a = a.next
                b = b.next
            else:
                b = b.next
        if not a and not b:
            return temp
        a = a.next
    return None


def findPeakElement_162(nums: List[int]) -> int:
    """
    顺序查找
    :param nums:
    :return:
    """
    i = 0
    if len(nums) == 1:
        return 0
    while i < len(nums):
        if (i == 0 and nums[i] > nums[i + 1]) or (
                i == len(nums) - 1 and nums[i] > nums[i - 1]) or (nums[i - 1] < nums[i] and nums[i] > nums[i + 1]):
            return i
        i += 1
    return 0


def findPeakElement_162_1(nums: List[int]) -> int:
    """
    二分法，
    :param nums:
    :return:
    """
    res = []
    if len(nums) == 1:
        return 0

    def _search(left, right):
        if left > right:
            return
        mid = (left + right) >> 1
        if (mid == 0 and nums[mid] > nums[mid + 1]) or (
                mid == len(nums) - 1 and nums[mid] > nums[mid - 1]) or (
                nums[mid - 1] < nums[mid] and nums[mid] > nums[mid + 1]):
            res.append(mid)
            return
        else:
            _search(left, mid - 1)
            _search(mid + 1, right)

    _search(0, len(nums) - 1)
    if res:
        return res[0]
    else:
        return None


def findPeakElement_162_2(nums: List[int]) -> int:
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = (left + right) >> 1
        if (mid == 0 and nums[mid] > nums[mid + 1]) or (
                mid == len(nums) - 1 and nums[mid] > nums[mid - 1]) or (
                nums[mid - 1] < nums[mid] and nums[mid] > nums[mid + 1]):
            return mid
        elif nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left


def twoSum_167(numbers: List[int], target: int) -> List[int]:
    left = 0
    right = len(numbers) - 1
    if numbers[0] + numbers[-1] < target:
        return []
    while left < right:
        if numbers[left] + numbers[right] == target:
            return [left + 1, right + 1]
        elif numbers[left] + numbers[right] < target:
            left += 1
        else:
            right -= 1
    return []


def NotAdjacentLine_1(nums):
    """
    给一个正整数list，找到和最大的序列，条件是序列不能相邻；
    :param nums:
    :return:
    """

    def _solution(nums, index):
        if index < 0:
            return -1
        if index == 0:
            return nums[0]
        return max(_solution(nums, index - 1), _solution(nums, index - 2) + nums[index])

    return _solution(nums, len(nums) - 1)


def NotAdjacentLine_2(nums):
    """

    dp[i] = max(dp[i-2]+nums[i], dp[i-1])
    :param nums:
    :return:
    """
    if not nums:
        return 0
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[-1]


def rob_213(nums: List[int]) -> int:
    """
    给一个正整数list，找到和最大的序列，条件是序列不能相邻；，如果数组是相邻的，
    因为是正整数，且是相邻，所以nums[0] 和 nums[1]不能同时选。

    :param nums:
    :return:
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    def _max_rob(nums):
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        return dp[-1]

    return max(_max_rob(nums[1:]), _max_rob(nums[:-1]))


def convertToTitle_168(n: int) -> str:
    if not n:
        return ''
    div = n
    result = []
    while div:
        div -= 1
        mod = div % 26
        div = div // 26
        result.insert(0, mod)
    # 数字转字符
    res = ''
    for num in result:
        res += chr(num + 65)
    return res


def majorityElement(nums: List[int]):
    # counter = collections.Counter(nums)
    # mc = counter.most_common()[0][0]
    return sorted(nums)[len(nums) // 2]


def majorityElement_169(nums: List[int]):
    # 摩尔投票法
    # candi永远表示最大相对票数的候选元素，candi_count对应其相对票数
    candi, candi_count = nums[0], 1
    for num in nums:
        # 表示直到当前num，并没有候选元素，就是没有一个元素的票数>0的
        if candi_count == 0:
            candi = num
            candi_count = 1
            continue
        # 当前num等于候选元素，则候选元素增加一票
        if candi == num:
            candi_count += 1
        else:
            candi_count -= 1  # 当前num不等于候选元素，并且最大票数大于0，只需要把最大票数减一，候选元素不变

    return candi


def titleToNumber_171(c: str) -> int:
    dup = []
    for char in c:
        dup.insert(0, ord(char) - 64)
    res = 0
    for index, num in enumerate(dup):
        res += num * (math.pow(26, index))
    return int(res)


def trailingZeroes_172(n: int) -> int:
    """
    统计5个个数
    :param n:
    :return:
    """
    five_num = 0
    while n >= 5:
        five_num += n // 5
        n //= 5
    return five_num


class BSTIterator:

    def __init__(self, root: TreeNode):
        self.root = root
        self.cur_node = 0
        self.order_values = []
        self._build_order(self.root, self.order_values)

    def _build_order(self, root, res):
        if not root:
            return
        self._build_order(root.left, res)
        res += [root.val]
        self._build_order(root.right, res)

    def _build(self):
        """
        用栈实现
        :return:
        """
        res = []
        stack = []
        cur = self.root
        while cur or stack:
            # 全部左孩子入栈
            while cur:
                stack.append(cur)
                cur = cur.left
            # 栈顶元素出栈，输出，并向右走
            cur = stack.pop()
            res += [cur.val]
            cur = cur.right
        return res

    def next(self) -> int:
        """
        @return the next smallest number
        """
        res = self.order_values[self.cur_node]
        self.cur_node += 1
        print(res)
        return res

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        res = self.cur_node < len(self.order_values)
        print(res)
        return res


class BSTIterator1:

    def __init__(self, root: TreeNode):
        self.ahead_node = []
        while root:
            self.ahead_node.append(root)
            root = root.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        cur = self.ahead_node.pop()
        res = cur.val
        cur = cur.right
        while cur:
            self.ahead_node.append(cur)
            cur = cur.left
        print(res)
        return res

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        res = self.ahead_node != []
        print(res)
        return res


if __name__ == '__main__':
    tree = BTree()
    tree.create_btree([7, 3, 15, None, None, 9, 20])

    # head = ListNode(2)
    # n1 = ListNode(2)
    # n2 = ListNode(4)
    # n3 = ListNode(5)
    # n4 = ListNode(4)
    #
    # head.next = n1
    # n1.next = n2
    # n2.next = n3
    # n3.next = n4

    # res = trailingZeroes_172(30)
    # print(res)
    # res = []
    # tree.in_order(tree.root, res)
    # print(res)
    obj = BSTIterator(tree.root)
    param_1 = obj.next()
    param_2 = obj.next()
    param_3 = obj.hasNext()
    p4 = obj.next()
    p5 = obj.hasNext()
    p6 = obj.next()
    p7 = obj.hasNext()
    p8 = obj.next()
    p9 = obj.hasNext()
