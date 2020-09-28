"""
大根堆：
1. 完全二叉树
2. 父节点的值大于等于子节点的值（小根堆相反）

用list表示堆：
第一个数表示堆顶

"""


class MyHeapq():
    def __init__(self):
        self.heap = []

    """
    堆中增加新元素
    """

    def my_heappush(self, item: int):
        self.heap.append(item)
        self._siftdown(self.heap, 0, len(self.heap) - 1)

    # ”上浮“，最后一个新增加的元素向上更新. 向下”筛选“更小的元素
    def _siftdown(self, heap, startpos, pos):
        # 记下新怎加的节点值
        newitem = heap[pos]
        while pos > startpos:
            parent_pos = (pos - 1) >> 1  # 父节点
            parent = heap[parent_pos]
            # 如果父节点小于当前节点，把父节点赋值到当前节点
            if heap[pos] < parent:
                heap[pos] = parent
                pos = parent_pos
                continue
            break
        # 更新值
        heap[pos] = newitem

    """
    输出堆顶元素
    """

    def my_heappop(self, heap):
        lastelt = heap.pop()
        if heap:
            # 第一个元素是堆顶元素，返回
            returnitem = heap[0]
            # 最后一个元素放在堆顶位置，然后执行”下沉“
            heap[0] = lastelt
            self._siftup(heap, 0)
            return returnitem
        return lastelt

    # ”下沉“
    def _siftup(self, heap, pos):
        endpos = len(heap)
        startpos = pos
        # 下沉的元素
        newitem = heap[pos]
        # 默认赋给左孩子
        childpos = 2 * pos + 1
        while childpos < endpos:
            # 检查是否右孩子更小
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            # 将更小的孩子上移
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * childpos + 1
        # 此时pos为叶子节点，下沉的元素赋值给叶子节点
        heap[pos] = newitem
        # 再让新元素进行”上浮“
        self._siftdown(heap, startpos, pos)

    """
    输出堆顶元素后，调整剩余元素，使之成为一个新的堆
    方法：下沉
    """

    """ 
    由一个无序列表构建一个堆：
    从列表n/2个元素（完全二叉树最后一个节点）开是，直到第一个元素，依次进行”下沉“。
    """

    def my_heapify(self, x):
        # 构建小根堆
        n = len(x)
        for i in reversed(range(n // 2)):
            self._siftup(x, i)


if __name__ == '__main__':
    heap = MyHeapq()
    nums = [3,2,1,5,6,4]
    heap.my_heapify(nums)
    print(nums)
