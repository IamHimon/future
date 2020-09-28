#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/17 11:23
# @Author  : humeng


class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


"""
参考：https://www.cnblogs.com/songwenjie/p/8955856.html
"""


class BTree(object):
    def __init__(self):
        self.root = None
        self.nodes = []

    # 创建二叉树
    def create_btree(self, nums):
        """
        创建二叉树，None为空,
         父节点 = nums[i](i < len(nums) // 2 -1),左孩子 = nums[2*i+1], 右孩子=nums[2*i + 2]
        :param nums:
        :return:
        """
        for n in nums:
            if n is None:
                self.nodes.append(None)
            else:
                self.nodes.append(TreeNode(n))
        # 遍历所有父亲
        for i in range(len(self.nodes) // 2):
            if 2 * i + 1 < len(nums):
                self.nodes[i].left = self.nodes[2 * i + 1]
            if 2 * i + 2 < len(nums):
                self.nodes[i].right = self.nodes[2 * i + 2]
        self.root = self.nodes[0]

    def pre_order(self, root: TreeNode, res):
        """
        前序遍历
        :param res:
        :param root:
        :return:
        """
        if root is None:
            return
        res += [root.val]
        self.pre_order(root.left, res)
        self.pre_order(root.right, res)

    def pre_order_stack(self, root: TreeNode, res):
        """
        非递归实现先序遍历,利用栈的特征，先入后出。
        根节点先压栈，记录出栈结点值，右孩子先压栈，左孩子再压栈，知道栈为空
        :param root:
        :param res:
        :return:
        """
        if root is None:
            return
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur is not None:
                res += [cur.val]
            if cur.right is not None:
                stack.append(cur.right)
            if cur.left is not None:
                stack.append(cur.left)

    def in_order(self, root: TreeNode, res):
        """
        中序遍历
        :return:
        """
        if root is None:
            return
        self.in_order(root.left, res)
        res += [root.val]
        self.in_order(root.right, res)

    def in_order_ite(self, root: TreeNode):
        """
        非递归实现中序遍历, 借用栈
        当前节点的左孩子先全部入栈，直到没有左孩子，栈顶结点出栈，输出，然后往右孩子走
        :param root:
        :param res:
        :return:
        """
        res = []
        if root is None:
            return
        stack = []
        cur = root
        while stack or cur is not None:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res += [cur.value]
            cur = cur.right
        return res

    def _in_order_ite2(self, root):
        """
        借助栈，每次取出栈顶节点，然后右孩子先入栈，左孩子再入栈
        """
        res = []
        if not root:
            return []
        stack = [root]
        while stack:
            top = stack.pop()
            res += [top.val]
            if top.right:
                stack.append(top.right)
            if top.left:
                stack.append(top.left)
        return res

    def post_order(self, root: TreeNode, res):
        """
        后续遍历
        :return:
        """
        if root is None:
            return
        self.post_order(root.left, res)
        self.post_order(root.right, res)
        res += [root.val]

    def post_order_it(self, root: TreeNode):
        """
        后续遍历非递归，
        用两个stack实现，后序遍历是：[左->右->父]
        在栈中是先进后出，则需要按照"父", [“左”,"右"]的顺序进stack1，从stack1中pop，然后依次进stack2就是[父，右，左]
        :return:
        """
        if root is None:
            return
        res = []
        stack1 = [root]
        stack2 = []
        while stack1:
            cur = stack1.pop()
            if cur.left:
                stack1.append(cur.left)
            if cur.right:
                stack1.append(cur.right)
            stack2.append(cur)
        while stack2:
            res.append(stack2.pop().value)
        return res

    def level_order(self, root: TreeNode):
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

    def level_order_ite(self, root: TreeNode):
        """
        递归解决
        :param root:
        :return:
        """
        if root is None:
            return []
        res = []

        def _level(root, level):
            if not root:
                return
            if len(res) == level:
                res.append([])
            res[level].append(root.val)
            if root.left:
                _level(root.left, level + 1)
            if root.right:
                _level(root.right, level + 1)

        _level(root, 0)
        return res


if __name__ == '__main__':
    tree = BTree()
    tree.create_btree([1, 2, 3, 4, 5, 6, 7])
    pre_result = []
    # tree.pre_order(tree.root, pre_result)
    # pre_result1 = []
    # print("pre_order:    ", pre_result)
    # tree.pre_order_stack(tree.root, pre_result1)
    # print("pre_order_ite:", pre_result1)
    in_result = []
    # tree.in_order(tree.root, in_result)
    # print("in_order:    ", in_result)
    # in_result1 = tree.in_order_ite(tree.root)
    # print("in_order_ite:", in_result1)
    # post_result = []
    # tree.post_order(tree.root, post_result)
    # print("post_order:     ", post_result)
    # post_result1 = tree.post_order_it(tree.root)
    # print("post_order_ite: ", post_result)

    level_result1 = tree.level_order_ite(tree.root)
    print("post_order_ite: ", level_result1)
