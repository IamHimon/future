from base_classes import *


def searchInsert1(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return 0
    start = 0
    end = len(nums)
    t = 0
    while end > start:
        t = start + (end - start) // 2
        if nums[t] == target:
            return t
        elif nums[t] < target:
            start = t + 1
        else:
            end = t
    return end


def countAndSay(n):
    """
    :type n: int
    :rtype: str
    """

    if n == 1:
        return '1'
    base = countAndSay(n - 1)
    say = ''
    i = 0
    while i < len(base):
        if i + 1 >= len(base) or base[i + 1] != base[i]:
            say += "1%s" % (base[i])
            i += 1
        else:
            j = i + 1
            while j < len(base) and base[j] == base[i]:
                j += 1
            say += "%s%s" % (j - i, base[i])
            i = j
    return say


def search(nums, target):
    """
    33，
    双指针
    """
    low = 0
    high = len(nums) - 1
    while high >= low:
        print("low:%s, high:%s" % (low, high))
        if nums[low] != target and nums[high] != target:
            low += 1
            high -= 1
        elif nums[low] > target and nums[high] < target:
            return -1
        elif nums[low] == target:
            return low
        elif nums[high] == target:
            return high
    return -1


# def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
# 	result = []
# 	sorted_nums = sorted(candidates)
# 	def _search(temp_res: List[int], target: int, index : int):
# 		if target == 0:
# 			return result.append(temp_res)
# 		if target < 0 or index >= len(sorted_nums):
# 			return
# 		if index < len(sorted_nums) and target > 0 and target >= sorted_nums[index]:
# 			temp_res.append(sorted_nums[index])
# 			target -= sorted_nums[index]
# 			_search(temp_res, target, index + 1)
# 	_search([], target, 0)
# 	return result


def multiply(num1: str, num2: str) -> str:
    """
    43,
    竖式乘法实现
    """

    def _split_mul(num, cur):
        result = []
        for n in num[::-1]:
            result.append(int(n) * cur)
        return result

    def _sum_bit(mul_res):
        # 进位加，[15,12,3] => 15 + 12*10 + 3 * 100
        res = 0
        for i, m_r in enumerate(mul_res):
            temp = (10 ** i) * m_r
            res += temp
        return res

    mul_res = []
    for n in num2[::-1]:
        mul_res.append(_split_mul(num1, int(n)))

    split_res = []
    for m_r in mul_res:
        split_res.append(_sum_bit(m_r))

    res = _sum_bit(split_res)
    return str(res)


from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    # 46, 回朔
    length = len(nums)
    set = ()
    result = []

    def _count(temp: List[int]) -> List[int]:
        if len(temp) == length:
            result.append(temp)
            return
        for num in nums:
            if num not in temp:
                _count(temp + [num])

    for num in nums:
        _count([num])

    return result


def permute3(self, nums: List[int]) -> List[List[int]]:
    # 46, 回朔
    res = []

    def backtrack(nums, tmp):
        if not nums:
            res.append(tmp)
        return
        for i in range(len(nums)):
            backtrack(nums[:i] + nums[i + 1:], tmp + [nums[i]])

    backtrack(nums, [])
    return res


def permuteUnique(nums: List[int]) -> List[List[int]]:
    # 47,
    result = []

    def _backtrack(flag: List[int], temp: List[int], index: int) -> List[int]:
        if any(flag):
            result.append(temp)
            return
        index = flag.index(0)
        if index:
            flag[index] = 1
            _backtrack(flag, temp + [nums[index]], index)

    flag = [0] * len(nums)
    for index in range(len(nums)):
        flag[index] = 1
        _backtrack(flag, [], index)

    return result


def _print(matrix):
    for line in matrix:
        print(line)


def rotate(matrix: List[List[int]]) -> None:
    """
    48.
    Do not return anything, modify matrix in-place instead.
    """

    N = len(matrix[0])

    def _rotate_round(start, end):
        for j in range(start, end):
            print("start:%s, j:%s" % (start, j))
            print('[%s, %s, %s, %s]' % (
                matrix[start][j], matrix[j][N - start - 1], matrix[N - start - 1][N - j - 1], matrix[N - j - 1][start]))
            print('-------------------------')
            _print(matrix)
            print('-------------------------')
            temp = matrix[start][j]
            matrix[start][j] = matrix[N - j - 1][start]
            matrix[N - j - 1][start] = matrix[N - start - 1][N - j - 1]
            matrix[N - start - 1][N - j - 1] = matrix[j][N - start - 1]
            matrix[j][N - start - 1] = temp
            _print(matrix)
            print("========================================")

    for start in range(N):
        if N - 2 * start - 1 <= 0:
            return
        _rotate_round(start, N - start - 1)


def rotate1(matrix: List[List[int]]) -> None:
    """
   48.
   Do not return anything, modify matrix in-place instead.
   现转置再翻转
   """


# for i in range(len(matrix[0])):
# 	for j in range(i):
# 		matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
# for row in matrix:
# 	row.reverse()


def groupAnagrams1(strs: List[str]) -> List[List[str]]:
    """
    49.

    """
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']
    temp_dict = {}

    def _get_index(one_str):
        temp_idx = []
        for c in one_str:
            temp_idx.append(chars.index(c))
        temp_idx.sort()
        key = ''.join(str(i) for i in temp_idx)
        if key in temp_dict:
            temp_dict[key] += [one_str]
        else:
            temp_dict[key] = [one_str]

    for one_str in strs:
        _get_index(one_str)

    return list(temp_dict.values())


def groupAnagrams1(strs: List[str]) -> List[List[str]]:
    # 49
    temp_dict = {}
    for one in strs:
        key = ''.join(sorted(one))
        if key in temp_dict:
            temp_dict[key].append(one)
        else:
            temp_dict[key] = [one]
    return list(temp_dict.values())


def myPow(x: float, n: int) -> float:
    # 50, 暴力法，超时
    if x == 0.0:
        return 0.0
    res = 1
    for i in range(abs(n)):
        res *= x
    if n >= 0:
        return res
    else:
        return 1 / res


def myPow1(x: float, n: int) -> float:
    # 50,快速幂
    res = 1

    def _pow(x, n):
        if n == 1:
            return x
        if n % 2 == 0:
            return _pow(x, n / 2) * _pow(x, n / 2)
        else:
            return _pow(x, n - 1) * x

    res = _pow(x, abs(n))
    if n < 0:
        return 1 / res
    else:
        return res


def myPow2(x: float, n: int) -> float:
    # 50,快速幂
    """
    递归模版：
    T fun(T n):
        if (边界条件1)：
            return T的值1
        elif(边界条件1):
            return T的值2
        else:
            T obj = fun(更小的事件)
            return obj

    """
    res = 1

    def _pow(x, n):
        if n == 0:
            return 1
        half = _pow(x, n // 2)
        if n % 2 == 0:
            return half * half
        else:
            return half * half * x

    if n < 0:
        x = 1 / x
        n = -n
    res = _pow(x, n)
    return res


def myPow3(x: float, n: int) -> float:
    # 50,快速幂
    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n < 0:
        return myPow3(1 / x, -n)
    elif n == 2:
        return x * x
    elif n % 2 == 0:
        return myPow3(x * x, n // 2)
    elif n % 2:
        return myPow3(x * x, n // 2) * x


def minDistance(word1: str, word2: str) -> int:
    import numpy as np
    n1 = len(word1) - 1
    n2 = len(word2) - 1
    dp = np.zeros((n1, n2), dtype=np.int)

    def _dist(i, j):
        if i < 0 or j < 0:
            return
        if word1[i] == word2[j]:
            dp[i][j] = dp[i - 1][j - 1]
        else:
            return min(_dist(i - 1, j - 1), _dist(i, j - 1), _dist(i - 1, j - 1)) + 1

    _dist(n1, n2)
    print(dp)
    return dp[0][0]


def minDistance1(word1: str, word2: str) -> int:
    """
    72, 编辑距离
    状态定义：dp[i][j]表示word1的前i个字母转换成word2的前j个字母所使用的最少操作。
    状态转移：若当前字母相同，则dp[i][j] = dp[i - 1][j - 1];
             否则取增删替三个操作的最小值 + 1， 即:
            dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    首字母加上" "，这样dp[0][j]和dp[i][0]是有意义的，这样，dp[i][0]表示从" "到dp[i]的编辑距离
    """

    word1 = " " + word1
    word2 = " " + word2
    n1 = len(word1)
    n2 = len(word2)
    dp = [[0] * n2 for _ in range(n1)]

    for i in range(n2):
        dp[0][i] = i

    for j in range(n1):
        dp[j][0] = j

    for i in range(1, n1):
        for j in range(1, n2):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[-1][-1]


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    """
    54,
    一圈一圈打印
    考察，边界下标的考虑
    """
    if len(matrix) <= 0:
        return []
    result = []
    n = len(matrix) - 1
    m = len(matrix[0]) - 1

    def _get_round(num):
        # (num, num)-> (n-num, m-num)

        # 左->右
        for i in range(num, m - num + 1):
            result.append(matrix[num][i])

        # 上->下
        for i in range(num + 1, n - num + 1):
            result.append(matrix[i][m - num])

        # 右->左
        for i in range(m - num - 1, num - 1, -1):
            if n - num <= num:
                break
            result.append(matrix[n - num][i])

        # 下-> 上
        for i in range(n - num - 1, num, -1):
            if m - num <= num:
                break
            result.append(matrix[i][num])

    num = 0
    while num <= n - num and num <= m - num:
        _get_round(num)
        num += 1

    return result


def canJump(nums: List[int]) -> bool:
    """
    55.
    把这个问题想象成，成你每走到一个格子拿到固定的能量值（num[i]的数值），
    每走一步消耗一个能量值，然后看能不能到达目的地(nums的最后一个位置)的问题。

    """
    length = len(nums)
    anagy = 0
    for i in range(0, length):
        anagy = max(anagy - 1, nums[i])
        if anagy <= 0 and i < length - 1:
            return False
    return True


def lengthOfLastWord(s: str) -> int:
    """
    58
    """
    if s == "":
        return 0
    splits = s.strip().split(' ')
    return len(splits[-1])


def lengthOfLastWord1(s: str) -> int:
    """
    58
    """
    if s == "":
        return 0
    s = s.strip()
    length = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] != ' ':
            length += 1
        else:
            break
    return length


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    56, 先用快速排序，然后递归合并
    """
    if len(intervals) <= 1:
        return intervals

    def _partitin(start, end):
        pivot = start
        while start < end:
            while start < end and intervals[end][0] >= intervals[pivot][0]:
                end -= 1
            while start < end and intervals[start][0] <= intervals[pivot][0]:
                start += 1
            intervals[start], intervals[end] = intervals[end], intervals[start]
        intervals[start], intervals[pivot] = intervals[pivot], intervals[start]
        return start

    def _sort(start, end):
        if start >= end:
            return
        mid = _partitin(start, end)
        _sort(start, mid - 1)
        _sort(mid + 1, end)

    def _merge(intervals, start):
        if start >= len(intervals) - 1:
            return
        if intervals[start][1] >= intervals[start + 1][0] or intervals[start][1] >= intervals[start + 1][1]:
            intervals[start] = [intervals[start][0], max(intervals[start + 1][1], intervals[start][1])]
            del intervals[start + 1]
            _merge(intervals, start)

        _merge(intervals, start + 1)

    _sort(0, len(intervals) - 1)
    _merge(intervals, 0)
    return intervals


def rotateRight(head: ListNode, k: int) -> ListNode:
    """
    61,
    定义“前进”：将最后一个结点放在开头
    k=1，相当于“前进”一次，k=2,“前进”2次，，如果head的长度为L，当k大于L，就是相当于“前进”了 k % L次
    """

    def _forward(head1):
        # "前进"一次
        tail = head1
        former = head1
        while tail.next is not None:
            former = tail
            tail = tail.next
        former.next = None
        tail.next = head1.next
        head1.next = tail

    first = ListNode(-1)
    first.next = head
    length = 0
    if length <= 1:
        return head
    tag = head
    while tag is not None:
        tag = tag.next
        length += 1
    step = k % length
    for i in range(step):
        _forward(first)

    return first.next


def rotateRight1(head: ListNode, k: int) -> ListNode:
    """
	61, 官方解法， 
	1. 将链表变成循环链表
	"""
    if not head: return None
    length = 1
    tail = head
    while tail.next is not None:
        tail = tail.next
        length += 1

    step = k % length

    if length <= 1 and step == 0:
        return head

    tail.next = head
    new_tail = head

    i = 0
    while i < length - step - 1:
        i += 1
        new_tail = new_tail.next

    new_head = new_tail.next
    new_tail.next = None
    return new_head


def addBinary_my(a: str, b: str) -> str:
    """
    67.
    """

    length = max(len(a), len(b))
    if len(a) < length:
        a = '0' * (length - len(a)) + a
    if len(b) < length:
        b = '0' * (length - len(b)) + b
    sum_list = []
    append = 0
    for i in range(length - 1, -1, -1):
        temp_sum = int(a[i]) + int(b[i]) + append
        if temp_sum >= 2:
            append = 1
            sum_list.insert(0, str(temp_sum % 2))
        else:
            sum_list.insert(0, str(temp_sum))
            append = 0
    if append == 1:
        sum_list.insert(0, '1')
    return ''.join(sum_list)


def addBinary(a: str, b: str) -> str:
    """

    :param a:
    :param b:
    :return:
    """
    a_len = len(a)
    b_len = len(b)
    append = 0
    sum_list = []
    for i in range(1, max(a_len, b_len) + 1):
        temp_sum = append
        if i <= a_len:
            temp_sum += int(a[-i])
        if i <= b_len:
            temp_sum += int(b[-i])

        if temp_sum >= 2:
            append = 1
            sum_list.insert(0, str(temp_sum % 2))
        else:
            sum_list.insert(0, str(temp_sum))
            append = 0

    if append == 1:
        sum_list.insert(0, '1')
    return ''.join(sum_list)


def addBinary1(a, b) -> str:
    x, y = int(a, 2), int(b, 2)
    while y:
        answer = x ^ y
        carry = (x & y) << 1
        x, y = answer, carry
    return bin(x)[2:]


def numDecodings_my(s: str) -> int:
    """
    91.
    动态规划,自顶向下,超时
    """

    def _call(s, pos):
        if pos >= len(s):  # 最后状态
            return 1
        count = 0
        if int(s[pos:pos + 1]) >= 1 and s[pos:pos + 1] != '0':
            count += _call(s, pos + 1)
        if pos + 2 <= len(s) and int(s[pos: pos + 2]) <= 26 and s[pos] != '0':
            count += _call(s, pos + 2)
        return count

    if s == '0':
        return 0
    count = _call(s, 0)

    return count


def numDecodings(s: str) -> int:
    """
    91. 类似爬楼梯，一次一级或者两极，只是需要考虑待“0”的特殊情况，需要进行拆分。
    动态规划,自低向上，
    定制：dp[i]表示，到i位置最多的解码个数
    状态方程：dp[i] = dp[i-1] + dp[i-2]
    """
    dp = [0] * len(s)
    # 初始化
    # 考虑第一个字母为0
    if s[0] == '0':
        return 0
    else:
        dp[0] = 1
    if len(s) == 1:
        return dp[-1]
    # 考虑第二个字母
    if s[1] != '0':
        dp[1] += 1
    if '10' <= s[:2] <= "26":
        dp[1] += 1

    # 从第二个开始
    for i in range(2, len(s)):
        # 考虑单个字母
        if s[i] != '0':
            dp[i] += dp[i - 1]
        # 考虑两个字母
        if s[i - 1: i + 1] == '00':
            return 0
        if '10' <= s[i - 1:i + 1] <= "26":
            dp[i] += dp[i - 2]
    return dp[-1]


def restoreIpAddresses(s: str) -> List[str]:
    """
    93. 回朔(画图)，
    剪枝：
    1. 最多4组，也就是最多4层树
    2. 每段位数在1-3之间
    3. ip格式每个部分占8位，就是各组数字小于255
    3. 当每段长度>1时，第一位不能是'0'；
    """
    result = []
    length = len(s)

    def _split(result, pos, s, ips):
        if pos == length and len(ips) == 4:
            result.append('.'.join(ips))
            return result
        # 取后面1位，
        if pos + 1 <= length and (4 - len(ips) - 1) * 3 >= length - pos - 1:
            _split(result, pos + 1, s, ips + [s[pos:pos + 1]])
        # 取后面2位
        if pos + 2 <= length and (4 - len(ips) - 1) * 3 >= length - pos - 2 and s[pos] != '0':
            _split(result, pos + 2, s, ips + [s[pos:pos + 2]])
        # 取后面3位
        if pos + 3 <= length and (s[pos:pos + 3] <= '255') and (4 - len(ips) - 1) * 3 >= length - pos - 3 and s[
            pos] != '0':
            _split(result, pos + 3, s, ips + [s[pos:pos + 3]])
        return result

    result = _split(result, 0, s, [])
    return result


def reverseWords_my(s: str) -> str:
    """
    151, 使用python api, strip, lstrip, split, reversed(迭代器)
    """
    s = s.strip().lstrip()
    splits_words = s.split(' ')
    splits_words = [word.strip() for word in splits_words if word != '']
    reverse_words = reversed(splits_words)
    return ' '.join(reverse_words)


def reverseWords(s: str) -> str:
    """
    151, 双指针
    """
    if s == ' ':
        return ''
    if len(s) == 1:
        return s
    result = []
    start = end = len(s) - 1
    while start >= 0 and end >= 0:
        if s[start] == ' ' and s[end] == ' ':
            start -= 1
            end -= 1
            continue
        if start == 0 and s[start] != ' ':
            result.append(s[start: end + 1])
        if s[start] != ' ':
            start -= 1
            continue
        if s[start] == ' ':
            result.append(s[start + 1: end + 1])
            start -= 1
            end = start

    return ' '.join(result)


def numTrees(n: int) -> int:
    """
    96,回朔无法解决！
    """
    if n <= 1:
        return n
    count = 0

    ranges = list(range(1, n + 1))
    all_nums = []

    def _search(nums, cur, father, count, all_nums):
        if len(nums) == n:
            count.append(1)
            all_nums.append(nums)
            return
        below_nums = [n for n in ranges if n not in nums and n < cur]
        above_nums = [n for n in ranges if n not in nums and n > cur]

        # 加双子
        if len(below_nums) and len(above_nums):
            for i in range(len(below_nums)):
                for j in range(len(above_nums)):
                    _search(nums + [below_nums[i], above_nums[j]], cur, above_nums[j], count, all_nums)

        # 加左子,小于父亲
        for m in below_nums:
            if m < father:
                _search(nums + [-m], m, cur, count, all_nums)
        # 加右子，大于父亲
        for m in above_nums:
            if m > father:
                _search(nums + [m * 10], m, cur, count, all_nums)
        return

    count = []
    for i in range(1, n + 1):
        _search([i], i, i, count, all_nums)
    print(count)
    print(all_nums)
    return len(count)


def numTrees1(n: int) -> int:
    """
    96, 卡特兰数,二叉搜索树的个数与分支的数量有关（序列的长度），与分支的内容无关(序列的数值)
    动态规划，
    假设n个结点存在，
    定义G(n)表示从1~n可以构成的二叉搜索（BST）树个数
    f(i):表示以结点i为跟的二叉搜索树个数，则G(n) = f(1) + ....f(n);
    当i为跟结点是，左边结点为：[1,2,,,i -1], 右边结点为：[i+1, ,,n-i],
    则左边有i-1个结点，共有G(i-1)个BST,右边有n-i个结点，共有G(n-i)个BST, 则f(i)=G(i-1)*G(n-i)
    则，G(n) = G(1)*G(n-1) + G(2)*G(n-2) + ... + G(n-1)*G(1)
    :param n:
    :return:
    """
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - j - 1]
    return dp[-1]


def inorderTraversal(root: TreeNode) -> List[int]:
    """
    94， 二叉树中序遍历
    :param self:
    :param root:
    :return:
    """

    def _it(root, result):
        if root is None:
            return result
        result.append(root.val)
        inorderTraversal(root.right)
        inorderTraversal(root.left)

    result = []
    _it(root, result)
    return result


def grayCode(n: int) -> List[int]:
    """
    89

    关键是搞清楚格雷编码的生成过程, G(i) = i ^ (i/2);
    如 n = 3:
    G(0) = 000,
    G(1) = 1 ^ 0 = 001 ^ 000 = 001
    G(2) = 2 ^ 1 = 010 ^ 001 = 011
    G(3) = 3 ^ 1 = 011 ^ 001 = 010
    G(4) = 4 ^ 2 = 100 ^ 010 = 110
    G(5) = 5 ^ 2 = 101 ^ 010 = 111
    G(6) = 6 ^ 3 = 110 ^ 011 = 101
    G(7) = 7 ^ 3 = 111 ^ 011 = 100

    """
    G = []
    num = 1 << n
    for i in range(num):
        G.append(i ^ (i >> 1))
    return G


def setZeroes_my(matrix: List[List[int]]) -> None:
    """
    73, 记录下0的下标,O(m+n)
    其他思路，把为0的列和行都设置为None
    """
    m = len(matrix)
    if m < 0:
        return

    n = len(matrix[0])

    def _change(matrix, i, j):
        if matrix[i][j] == 0:
            for m_i in range(m):
                matrix[m_i][j] = 0
            for n_j in range(n):
                matrix[i][n_j] = 0

    all_zero_indexes = []
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                all_zero_indexes.append((i, j))
    print(all_zero_indexes)
    for i, j in all_zero_indexes:
        _change(matrix, i, j)


def setZeroes(matrix: List[List[int]]) -> None:
    """
    73, 先把0替换成None
    """
    m = len(matrix)
    if m < 0:
        return

    def _change(matrix, i, j):
        matrix[i][j] = None
        for m_i in range(m):
            if matrix[m_i][j] != 0:
                matrix[m_i][j] = None
        for n_j in range(n):
            if matrix[i][n_j] != 0:
                matrix[i][n_j] = None

    n = len(matrix[0])
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 0:
                _change(matrix, i, j)

    # 替换None
    for i in range(m):
        for j in range(n):
            if matrix[i][j] is None:
                matrix[i][j] = 0


def lengthOfLongestSubstring(s: str) -> int:
    """
    1. 最大子串,利用队列，遍历s, 如果当前char在队列中，则队列char所在位置前面全部出列,否则append.并使用max_length记录queue最大值
    """
    max_length = 0
    queue = []
    for char in s:
        if char not in queue:
            queue.append(char)
        else:
            first = queue[0]
            while first != char:
                del queue[0]
                if len(queue) > 0:
                    first = queue[0]
                else:
                    first = ''
            del queue[0]
            queue.append(char)
        max_length = max(len(queue), max_length)
    return max_length


def uniquePaths(m: int, n: int) -> int:
    """
    62,  动态规划,
    定义：dp[i][j]为到达(i,j)位置最多的路径
    方程： dp[i][j] = dp[i-1][j] + dp[i][j-1] (=左边+上面)

    """

    dp = [[1] * m] + [[1] + [0] * (m - 1) for _ in range(1, n)]
    for n_i in range(1, n):
        for m_i in range(1, m):
            dp[n_i][m_i] = dp[n_i - 1][m_i] + dp[n_i][m_i - 1]
    return dp[-1][-1]


def generateMatrix(n: int) -> List[List[int]]:
    """
    59, 生成螺旋数组,边界
    """
    # init matrix
    matrix = [[0] * n for _ in range(n)]

    def _gen_rounde(num, index, all_dig):
        # 生成一圈,从(num,num)开始
        # 左->右
        for i in range(num, n - num):
            if index < len(all_dig):
                matrix[num][i] = all_dig[index]
                index += 1
        # 上-> 下
        for i in range(num + 1, n - num):
            if index < len(all_dig):
                matrix[i][n - num - 1] = all_dig[index]
                index += 1
        # 右->左
        for i in range(n - num - 2, num - 1, -1):
            if index < len(all_dig):
                matrix[n - num - 1][i] = all_dig[index]
                index += 1
        # 下 -> 上
        for i in range(n - num - 2, num, -1):
            if index < len(all_dig):
                matrix[i][num] = all_dig[index]
                index += 1
        return index

    all_dig = [i for i in range(1, pow(n, 2) + 1)]
    num = 0
    dig_index = 0
    while num < n - num:
        print('index: %s, num : %s' % (dig_index, num))
        dig_index = _gen_rounde(num, dig_index, all_dig)
        num += 1
    return matrix


def sortColors(nums: List[int]) -> None:
    """
    75.
    Do not return anything, modify nums in-place instead.
    快速排序的思想
    """

    def _partitin(start, end, nums):
        pivot = start
        while start < end:
            while nums[end] >= nums[pivot] and start < end:
                end -= 1
            while nums[start] <= nums[pivot] and start < end:
                start += 1
            nums[start], nums[end] = nums[end], nums[start]
        nums[pivot], nums[end] = nums[end], nums[pivot]
        return start

    def _ite(nums, start, end):
        if start >= end:
            return
        mid = _partitin(start, end, nums)
        _ite(nums, start, mid - 1)
        _ite(nums, mid + 1, end)

    _ite(nums, 0, len(nums) - 1)


def sortColors1(nums: List[int]) -> None:
    """
    75.
    Do not return anything, modify nums in-place instead.
    三指针
    begin:0区间的下一个位置，初始：0
    cur:当前位置,初始：0
    end:2区间的前一个位置，初始：最后一个位置
    分析：
    前进cur指针，
    如果cur是0，begin是1，则交换cur和begin
    如果cur是2，交换cur和end，并且end前移，cur不变
    cur是1或者cur是0，begin也是0，只前进cur

    """
    begin = cur = 0
    end = len(nums) - 1
    while cur <= end:
        if nums[cur] == 0:
            if nums[begin] == 1:
                nums[begin], nums[cur] = nums[cur], nums[begin]
                begin += 1
                cur += 1
            else:
                cur += 1
                begin += 1
        elif nums[cur] == 1:
            cur += 1
        elif nums[cur] == 2:
            nums[cur], nums[end] = nums[end], nums[cur]
            end -= 1


def combine(n: int, k: int) -> List[List[int]]:
    """
    77. 组合数组
    回朔算法
    """
    result = []

    def _gen(result, com, i):
        """

        :param result:
        :param com: 一个组合
        :param i:  当前取数下标
        :return:
        """
        if len(com) == k:  # 控制数高度
            result.append(com)
            return

        for num in range(i + 1, n + 1):
            _gen(result, com + [num], num)  # 从下一个数开始取
        return

    for i in range(1, n - k + 2):
        _gen(result, [i], i)

    return result


def subsets_78(nums: List[int]) -> List[List[int]]:
    """
    78 借助combinations解决
    """
    result = []
    for i in range(len(nums) + 1):
        from itertools import combinations
        for com in combinations(nums, i):
            result.append(list(com))
    return result


def subsets_78_1(nums: List[int]) -> List[List[int]]:
    """
    78. 回朔方法,实际上就是树的前序遍历，
    回朔三要素：
    1. 有效结果
        没有条件，都满足
    2. 回溯范围及答案更新
        需要循环遍历，而且从当前位置下一个位置开始
    3. 剪枝条件
        不需要剪枝

    """
    result = []
    n = len(nums)

    def _get_com(com, k):
        result.append(com)
        for i in range(k, n):
            _get_com(com + [nums[i]], i + 1)

    _get_com([], 0)
    return result


def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    """
    74, 两次二分法解决
    """

    def _binary_search(nums, target, left, right):
        if left > right:
            return False
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        if nums[mid] > target:
            return _binary_search(nums, target, left, mid - 1)
        elif nums[mid] < target:
            return _binary_search(nums, target, mid + 1, right)

    column = -1
    top_col = 0
    base_col = len(matrix) - 1
    while top_col <= base_col:
        mid_col = (top_col + base_col) // 2
        if len(matrix[mid_col]) == 0:
            break
        if matrix[mid_col][-1] < target and matrix[mid_col][0] < target:
            top_col = mid_col + 1
        elif matrix[mid_col][-1] > target and matrix[mid_col][0] > target:
            base_col = mid_col - 1
        else:
            column = mid_col
            break
    if column == -1:
        return False

    target_nums = matrix[column]

    res = _binary_search(target_nums, target, 0, len(target_nums) - 1)

    return res


def searchMatrix1(matrix: List[List[int]], target: int) -> bool:
    """
    74， 一次二分法解决问题
    """
    if len(matrix) == 0:
        return False
    left = 0
    right = len(matrix) * len(matrix[0]) - 1
    dim = len(matrix[0])
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // dim][mid % dim]
        if target == mid_value:
            return True
        else:
            if target > mid_value:
                left = mid + 1
            elif target < mid_value:
                right = mid - 1
    return False


def exist(board: List[List[str]], word: str) -> bool:
    """
    79, 类似八皇后问题
    回朔算法
    """
    if len(board) == 0 or "" == word:
        return False
    if len(board[0]) == 0:
        return False

    def _search(index, i, j):
        if index == len(word):
            return True

        if i < 0 or j < 0 or i == len(board) or j == len(board[0]):
            return False

        if board[i][j] != word[index]:
            return False
        # 记下当前值
        temp = board[i][j]

        # 当前位置置空，避免重复寻找
        board[i][j] = ''

        # 上
        if _search(index + 1, i - 1, j):
            return True
        # 下
        if _search(index + 1, i + 1, j):
            return True
        # 左

        if _search(index + 1, i, j + 1):
            return True
        # 右
        if _search(index + 1, i, j - 1):
            return True
        # 回朔
        board[i][j] = temp
        return False

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == word[0]:
                if _search(0, i, j):
                    return True
    return False


def removeDuplicates(nums: List[int]) -> int:
    """
    80, 滑窗遍历，递归地把重复的换到数组最后面。
    """

    def _change(nums, cur, cur_value):
        # 移到最后，
        i = cur
        while i < len(nums) - 1:
            nums[i] = nums[i + 1]
            i += 1
        nums[len(nums) - 1] = cur_value
        if nums[cur] == cur_value:
            _change(nums, cur, cur_value)

    if len(nums) <= 2:
        return len(nums)
    pre = 0
    cur = 1
    while cur < len(nums) and nums[cur] >= nums[pre]:
        # 滑窗内相同，并且下一个也相同
        if nums[cur] == nums[pre] and cur < len(nums) - 1 and nums[cur + 1] == nums[cur]:
            # 剩下的都一样，结束
            if nums[cur + 1] == nums[len(nums) - 1]:
                cur += 1
                break
            _change(nums, cur + 1, nums[cur + 1])
        cur += 1
        pre += 1
    return cur


def removeDuplicates2(nums: List[int]) -> int:
    if len(nums) <= 2:
        return len(nums)
    i = 0
    for num in nums:
        if i < 2 or num > nums[i - 2]:
            nums[i] = num
            i += 1
    return i


def search_81(nums: List[int], target: int) -> bool:
    """
    81,
    """
    if len(nums) == 0:
        return False

    def _binary_search(start, end):
        if start > end:
            return False
        mid = (start + end) // 2
        if nums[mid] == target:
            return True
        if nums[mid] < target:
            return _binary_search(mid + 1, end)
        elif nums[mid] > target:
            return _binary_search(start, mid - 1)

    # 找到pivot
    i = 0
    while i < len(nums) - 1 and nums[i + 1] >= nums[i]:
        i += 1

    return _binary_search(0, i) or _binary_search(i + 1, len(nums) - 1)


def deleteDuplicates_82(head: ListNode) -> ListNode:
    """
    82,Remove Duplicates from Sorted List II
    两个指针，然后新建一个数组
    :param head:
    :return:
    """
    if head is None:
        return None
    result = ListNode(-1)
    result_cur = result
    pre = head
    cur = pre.next
    step = 0
    while pre:
        if cur is not None and pre.val == cur.val:
            cur = cur.next
            step = 1
        else:
            if cur is None and step == 0:
                result_cur.next = ListNode(pre.val)
                result_cur = result_cur.next
                pre = cur
            elif step == 0:
                result_cur.next = ListNode(pre.val)
                result_cur = result_cur.next
                pre = cur
                cur = cur.next
            else:
                pre = cur
                if cur is not None:
                    cur = cur.next

            step = 0
    return result.next


def partition_86(head: ListNode, x: int) -> ListNode:
    """
    86,Partition List
    使用双指针，用一个链表暂存小于x的结点，把小于x的结点从原链表中去掉，最后再拼接两个链表
    """
    if head is None:
        return None
    if head.next is None:
        return head
    dummyhead = ListNode(-1)
    dummyhead1 = ListNode(-1)
    dummycur = dummyhead
    dummyhead1.next = head
    pre = dummyhead1
    cur = pre.next
    while cur:
        # 如果cur小于x，则加入新的链表中,pre指向cur.next
        if cur.val < x:
            dummycur.next = ListNode(cur.val)
            dummycur = dummycur.next
            pre.next = cur.next
            cur = cur.next
        else:
            cur = cur.next
            pre = pre.next
    # dymmycur指向剩下的大于x的结点
    dummycur.next = dummyhead1.next
    return dummyhead.next


def partition_86_2(head: ListNode, x: int) -> ListNode:
    """
    官方解法,写法更优雅一点，但是时间和空间都与自己的方法基本上一样。
    双指针，并创建两条链表，一个指向小于x的链表，一个指向大于等于x的链表
    :param head:
    :param x:
    :return:
    """
    after = after_head = ListNode(-1)
    before = before_head = ListNode(-1)
    while head:
        if head.val >= x:
            after.next = head
            after = after.next
        else:
            before.next = head
            before = before.next
        head = head.next
    # after.next = null 避免了生成环形链表
    after.next = None
    before.next = after_head.next
    return before_head.next


def subsetsWithDup_90(nums: List[int]) -> List[List[int]]:
    """
    90.Subsets II
    1. 先给nums排序，
    2. 使用回朔法找所有subset，实际上就是树的遍历，
        有效结果：当前subset不在result中（这是为什么要先排序）
    """
    result = []
    N = len(nums)
    nums.sort()

    def _subset(subset, k):
        if subset not in result:
            result.append(subset)
        for i in range(k, N):
            if i > k and nums[i] == nums[i - 1]:
                continue
            _subset(subset + [nums[i]], i + 1)
        return

    _subset([], 0)
    return result


def generateTrees_95(n: int) -> List[TreeNode]:
    """
    95. Unique Binary Search Trees II
    递归的方法，
    """

    def _gen_tree(start, end):
         # 此时没有数字，将null加入结果中
        if start > end:
            return [None, ]
        all_trees = []
        # 尝试每个数字作为根节点
        for i in range(start, end + 1):
            # 得到所有可能的左子树
            left_trees = _gen_tree(start, i - 1)
            # 得到所有可能的右子树
            right_trees = _gen_tree(i + 1, end)
            # 左子树右子树两两组合
            for l in left_trees:
                for r in right_trees:
                    cur_tree = TreeNode(i)
                    cur_tree.right = r
                    cur_tree.left = l
                    # 加入到最终结果中
                    all_trees.append(cur_tree)
        return all_trees

    return _gen_tree(1, n) if n else []





if __name__ == '__main__':
    # res = subsets1([1, 2, 3])
    # print(res)
    # matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]]
    # matrix = [[1]]
    # res = searchMatrix1(matrix, 3)
    #
    # board = [
    #     ['F', 'B', 'C', 'E'],
    #     ['S', 'F', 'C', 'F'],
    #     ['A', 'D', 'E', 'E']
    # ]
    # res = exist(board, 'FC')
    # head = ListNode(-1)
    # n1 = ListNode(1)
    # n2 = ListNode(4)
    # n3 = ListNode(3)
    # n4 = ListNode(2)
    # n5 = ListNode(5)
    # n6 = ListNode(2)
    # n7 = ListNode(5)
    # head.next = n1
    # n1.next = n2
    # n2.next = n3
    # n3.next = n4
    # n4.next = n5
    # n5.next = n6
    # n6.next = n7

    # nums = [2, 2, 2, 0, 1]
    # Input = [1,2,2]
    res = generateTrees_95(5)
    print(res)
