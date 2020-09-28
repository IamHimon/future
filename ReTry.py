from typing import List

from base_classes import ListNode, TreeNode


def maxSubArray(nums: List[int]) -> int:
    """
    最大字串之和
    动态规划
    定义：dp[i],点i位置之前的连续字串最大字串
    转移方程：dp[i] = max(dp[i-1], dp[i-1] + nums[i])
    初始化：dp[0] = nums[0]

    :param nums:
    :return:
    """
    L = len(nums)
    if not L:
        return 0
    dp = [0] * L
    dp[0] = nums[1]
    for i in range(1, L):
        dp[i] = max(dp[i - 1], dp[i - 1] + nums[i])
    print(dp)
    return dp[-1]


def deleteDuplicates_82(head: ListNode) -> ListNode:
    """
    三指针实现，如果cur与前继和后继都不相同，把cur加入新链表
    :param head:
    :return:
    """
    if not head:
        return None
    cur = head
    pre = ListNode(head.val - 1)
    pre.next = head
    new_link = ListNode(-1)
    nl = new_link
    while cur:
        if cur.val != pre.val and (cur.next is None or cur.val != cur.next.val):
            nl.next = ListNode(cur.val)
            nl = nl.next
        cur = cur.next
        pre = pre.next
    return new_link.next


def subsetsWithDup_90(nums: List[int]) -> List[List[int]]:
    if not nums:
        return []
    result = []
    L = len(nums)
    nums = sorted(nums)

    def _dfs(start, com):
        if start > L:
            return

        result.append(com)

        for i in range(start, L):
            # 在start之后，有重复的，只选择一个，比如[1,2,2,2,2],start =1,后面的，只需要取一个2就行了
            if i > start and nums[i] != nums[i - 1]:
                continue
            _dfs(i + 1, com + [nums[i]])

    _dfs(0, [])
    return result


def buildTree_105(preorder: List[int], inorder: List[int]) -> TreeNode:
    if not preorder or not inorder:
        return None

    root = TreeNode(preorder[0])
    in_index = inorder[preorder[0]]
    root.left = buildTree_105(preorder[1:in_index], inorder[:in_index])
    root.right - buildTree_105(preorder[:in_index], inorder[in_index + 1:])
    return root


def majorityElement_169(nums):
    """
    摩尔投票法
    """
    candi, candi_count = nums[0], 0
    for num in nums:
        if candi_count == 0:
            candi = num
            candi_count += 1
            continue
        if num == candi:
            candi_count += 1
        else:
            candi_count -= 1
    return candi


if __name__ == '__main__':
    # n1 = ListNode(1)
    # n2 = ListNode(1)
    # n3 = ListNode(3)
    # n4 = ListNode(4)
    # n5 = ListNode(4)
    # n6 = ListNode(6)
    # # head.next = n1
    # n1.next = n2
    # n2.next = n3
    # n3.next = n4
    # n4.next = n5
    # n5.next = n6
    res = majorityElement_169([1, 2, 2, 1, 3, 1, 4, 5, 1])
    print(res)
