"""
单调队列的实现(单调增)
"""


class MonotonicQueue(object):
    def __init__(self):
        # 时刻保持queue中item保持一定顺序
        self.queue = []

    def push(self, item):
        # 比item小的都出列
        while self.queue and self.queue[-1] < item:
            self.queue.pop()
        self.queue.append(item)

    def pop(self, item):
        # 如果刚好是最后一个，才出列
        if self.queue and item == self.queue[0]:
            return self.queue.remove(item)

    def max(self):
        if self.queue:
            return self.queue[0]
        else:
            return None
