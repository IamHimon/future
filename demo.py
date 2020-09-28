# s = sorted('asdfasdf')
# print(s)
# sorted(iterable, cmp=None, key=None, reverse=False)

# for i in range(10, 0, -1):
#     print(i)


# print(max('1', '0'))

# def lengthOfLongestSubstring(s: str) -> int:
# 	"""
# 	1. 最大子串,利用队列，遍历s, 如果当前char在队列中，则队列char所在位置前面全部出列,否则append.并使用max_length记录queue最大值
# 	"""
# 	max_length = 0
# 	queue = []
# 	for char in s:
# 		if char not in queue:
# 			queue.append(char)
# 		else:
# 			first = queue[0]
# 			while first != char:
# 				del queue[0]
# 				if len(queue) > 0:
# 					first = queue[0]
# 				else:
# 					first = ''
# 			del queue[0]
# 			queue.append(char)
# 		# print("char:%s, queue:%s"%(char, queue))
# 		max_length  = max(len(queue), max_length)
# 	return max_length

# if __name__ == '__main__':
#     res = lengthOfLongestSubstring("abcabcbb")
#     print(res)
# import math
# for i in range(pow(2, 2)):
# 	print(i)
# nums = [1,2,3]
# from itertools import combinationsPoly-encoder
# all_com = combinations(nums, 0)
# all_com = [list(com) for com in all_com]
# print(all_com)
# flag = [1,1]
# print(all(flag))
# nums = [[], [1], [1, 2], [1, 2, 2]]
# n = [1,2]
# print(n in nums)
# import TreeNode
#
#
# class Solution:
#     flag = False
#     def hasPathSum(self, root: TreeNode, target: int) -> bool:
#         if root is None:
#             return False
#
#         def _dfs(root, sum):
#             if sum == target and root.left is None and root.right is None:
#                 Solution.flag = True
#                 return
#             if root.left:
#                 _dfs(root.left, sum + root.left.val)
#             if root.right:
#                 _dfs(root.right, sum + root.right.val)
#             return
#
#         _dfs(root, root.val)
#         return Solution.flag

# for i in range(1, 6):
#    print(i)

# def _is_pal(s):
#     is_pal = True
#     for i in range(len(s) // 2):
#         print("i:%s,  j:%s, start:%s, end:%s"%(i, len(s) - 1 - i, s[i], s[len(s) - 1 - i]))
#         if s[i] != s[len(s) - 1 - i]:
#             is_pal = False
#             break
#     return is_pal
#
# print(_is_pal('aaaa'))

# print(0 ^ 3)
# s = '123456'
# print(s[::-1])

# # print(ord('a') - 96)

# import tensorflow as tf
# import numpy as np
#
# np.set_printoptions(suppress=True)
#
# temp_k = tf.constant([[10,0,0],
#                       [0,10,0],
#                       [0,0,10],
#                       [0,0,10]], dtype=tf.float32)  # (4, 3)
#
# temp_v = tf.constant([[   1,0],
#                       [  10,0],
#                       [ 100,5],
#                       [1000,6]], dtype=tf.float32)  # (4, 2)
#
# # 这条 `请求（query）符合第二个`主键（key）`，
# # 因此返回了第二个`数值（value）`。
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
#
# n = tf.shape(temp_k)[-1]
#

# data = tf.constant(np.arange(10).reshape(5, 2), dtype=tf.float32)
# layer = tf.keras.layers.LayerNormalization(axis=1)
# output = layer(data)
# print(data)
# print(output)
# with tf.Session() as sess:
#     print(sess.run(data))
#     print(sess.run(output))

# print(ord('Z') - 64)
# import time
# res1 = []
# res2 = set()
# t = 200
#
# for i in range(10000):
#     res1.append(i)
#     res2.add(i)
#
# t0 = time.time()
# print(t in res1)
# t1 = time.time()
# print(t1 - t0)
# print(t in res2)
# print(time.time() - t1)

# print(bin(43261596).replace('0b', ''))
# s = [1,2,3]
# print(s[::-1])
# print(-1//2)
# print(0//2)
# print(1//2)
# print(2//2)
# print(3//2)
# print(4//2)
# num = [1,2,None,None, 2,1]
# print(num)
# print(num[::-1])
# num.reverse()
# print(num)
# print(num == num[::-1])

# from heapq import heapify, nlargest
#
# #
# nums = [3, 2, 1, 5, 6, 4]
# heapify(nums)
# print(nums)
# res = nlargest(3, nums)
# print(res)
# s = nlargest(2, nums)
# print(s)


# print('1')

# print(3 * 2)
# print(int('3') * int('2'))
# n = [4715, 1285, 363, 124, 53, 16, 9]
# print(sum(n))
# m = [4715, 1285, 726, 372, 212, 90, 54]
# print(sum(m))
# print(2 //3)
class Solution:
    _max = 0
    def cuttingRope(self, n: int) -> int:
        if not n:
            return 0

        def _method(n, mul_res):
            if n <= 1:
                self._max = max(mul_res, self._max)
                return
            for i in range(1, n + 1):
                if n - i <= 0:
                    continue
                _method(n - i, mul_res * i)
            return

        _method(n, 1)
        return self._max