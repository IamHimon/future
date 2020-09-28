"""
记录链表相关的方法
目录：
1. 单链表排序(归并）
2. 单链表排序(快排）
"""

from base_classes import ListNode

"""
1.单链表排序(快排）
与快排思路一样，partition每次按照第一个节点未基准把链表分为前后两部分，然后返回中间节点，实现方式使用快慢指针交换节点值。
"""


def quick_sorted(head):
    def partition(head, end):
        """
        采用换val的方式实现:
        以头节点val为基准，把链表分排序，前面的小于val，后面的大于val
        用两个快（p1）慢（p2）指针实现，时刻保证p1之前都是小于pivot，p1和p2之间大于等于pivot，使用p2遍历指针，直到遇到end位置。
        :param head:
        :param end:
        :return:
        """
        pivot = head.val
        p1, p2 = head, head.next  # p2是遍历指针，p1是小数的指针
        while p2 != end:
            if p2.val < pivot:
                # 注意：这里需要交换大于pivot和小于pivot的两个指针值
                # 而这里要p1要先往前走一步，因为当前的p1是小于pivot的，p1.next才是大于pivot的！！
                p1 = p1.next
                # 交换p1和p2
                p1.val, p2.val = p2.val, p1.val
            p2 = p2.next
        # pivot放到最后一个小数位置
        head.val, p1.val = p1.val, head.val
        return p1

    def _ite(left, right):
        if left != right:
            mid = partition(left, right)
            _ite(left, mid)
            _ite(mid.next, right)

    # 把None当作最后一个节点
    _ite(head, None)


"""
2. 单链表排序(归并）
使用div和merge

"""


def merge_sorted(head):
    def _div(head):
        """
        递归分割
        :param head:
        :return:
        """
        # 为空，需要返回head
        if not head or not head.next:
            return head
        left_half, right_half = _split(head)
        # 因为div()返回的是合并的链表，所以这边需要给left_half和right_half重新赋值，这样才不会断链
        left_half = _div(left_half)
        right_half = _div(right_half)
        # merge 这里需要return
        return _merge(left_half, right_half)

    def _split(head):
        # 将链表从中间一份为二
        p1 = head
        p2 = head.next

        while p2 and p2.next:
            p1 = p1.next
            p2 = p2.next.next
        # 右边半边的头指针
        right = p1.next
        # 前后断开
        p1.next = None
        return head, right

    def _merge(left, right):
        """
        合并连个链表
        :param left:
        :param right:
        :return:
        """
        dummy = ListNode(-1)
        cur = dummy
        while left and right:
            if left.val < right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next
            cur = cur.next
        # 剩下left或者right继续添加到dummy
        if left:
            cur.next = left
        if right:
            cur.next = right
        return dummy.next

    return _div(head)


def print_node(head):
    if not head:
        return
    values = []
    while head:
        values.append(str(head.val))
        head = head.next
    print("->".join(values))


if __name__ == '__main__':
    # head = ListNode(-1)
    n1 = ListNode(5)
    n2 = ListNode(4)
    n3 = ListNode(7)
    n4 = ListNode(8)
    n5 = ListNode(1)
    n6 = ListNode(3)
    n7 = ListNode(6)
    # head.next = n1
    n1.next = n2
    n2.next = n3
    n3.next = n4
    n4.next = n5
    n5.next = n6
    n6.next = n7

    print_node(n1)
    new = merge_sorted(n1)
    print_node(new)
