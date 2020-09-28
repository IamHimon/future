#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/17 11:21
# @Author  : humeng

class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class ListNode:
    """
    链表
    """
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    """
    二叉树
    """
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None