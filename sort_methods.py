# 排序算法总结

# 1. 交换排序
#		1.1 冒泡排序
#		1.2 快速排序
# 2. 插入排序
# 		2.1 简单插入排序
# 		2.2 归并排序
#
# 3. 堆排序
# 4. 选择排序
#		4.1 简单选择排序
# 5. 归并排序
import copy


def bubbleSort(nums):
    # 冒泡排序
    # 注意内层边界是j-1
    # 时间复杂度：O(n*2),空间复杂度: O(1)
    if len(nums) <= 1:
        return nums
    i = 0
    j = len(nums)
    while i < j:
        for i in range(j - 1):
            if nums[i] > nums[i + 1]:
                nums[i + 1], nums[i] = nums[i], nums[i + 1]
        j -= 1
    return nums


def quickSort(nums):
    # 快速排序
    # 1.选取头位置作为”基准“，分别从尾部和头部往中间移动（先尾部），
    # 	从尾部看，循环找到小于基准的元素，从头部看，循环找到大于基准的元素，交换二者。返回尾部和头部重合的下标mid。
    # 2. 递归在mid左边和右边再进行上述操作。
    # 注意：partition函数中 移动先从尾向头，并返回mid的小标
    # 划分次数O(logn),每个分区遍历比较O(n),则时间复杂度：O(nlogn)。空间复杂度，每个分区需要记录一次基准，O(logn)

    def _partition(start, end):
        pivot_index = start
        while start < end:
            while start < end and nums[end] >= nums[pivot_index]:
                end -= 1
            while start < end and nums[start] <= nums[pivot_index]:
                start += 1
            (nums[end], nums[start]) = (nums[start], nums[end])
        (nums[pivot_index], nums[start]) = (nums[start], nums[pivot_index])
        return start

    def _ite(start, end):
        if start >= end:
            return
        mid = _partition(start, end)
        _ite(start, mid - 1)
        _ite(mid + 1, end)

    _ite(0, len(nums) - 1)

    return nums


def InsertSort(nums):
    # 直接插入排序
    # 从前往后移
    # 时间复杂度O(n^2)， 空间复杂度O(1)
    if len(nums) <= 1:
        return nums
    for i in range(1, len(nums)):
        j = i
        target = nums[i]
        while j > 0 and target < nums[j - 1]:
            nums[j] = nums[j - 1]  # 比较、后移，给target腾位置
            j -= 1
        nums[j] = target


def HeapSort(nums):
    # 堆排序
    # 
    # 参考：https://juejin.im/post/5bea6af051882548161b0f02
    def _up(end):
        i = end
        while i > 0:
            root = (i - 1) // 2
            if nums[root] < nums[i]:
                nums[root], nums[i] = nums[i], nums[root]
            i -= 1

    for e in range(len(nums) - 1, -1, -1):
        _up(e)
        nums[e], nums[0] = nums[0], nums[e]


def SelectSort(nums):
    # 简单选择排序
    length = len(nums)
    if length <= 1:
        return

    def _get_target(start):
        target = start
        for i in range(start, length):
            if nums[i] < target:
                target = i
        return target

    for i in range(length - 1):
        target = _get_target(i + 1)
        print("i: %s, target: %s" % (i, target))
        if nums[i] > nums[target]:
            nums[i], nums[target] = nums[target], nums[i]


def quick_sort(nums):
    def partition(left, right):
        pivot = nums[left]
        while left < right:
            # 从右边找第一个小于pivot的位置,
            while nums[right] >= pivot and left < right:
                right -= 1
            # 找到并放到前面
            nums[left] = nums[right]
            # 从左边找第一个大于pivot的位置
            while nums[left] <= pivot and left < right:
                left += 1
            # 找到并放到后面
            nums[right] = nums[left]

        # 分割点pivot赋值到left位置
        nums[left] = pivot
        return left

    def _ite(left, right):
        if left >= right:
            return
        mid = partition(left, right)
        _ite(left, mid - 1)
        _ite(mid + 1, right)

    _ite(0, len(nums) - 1)


def merge_sort(nums):
    """
    1. 分割：递归地把当前序列平均分割成两半。
    2. 集成：在保持元素顺序的同时将上一步得到的子序列集成到一起（归并）
    :param nums:
    :return:
    """

    def _div(nums, left, right):
        if left >= right:
            return
        # 递归分割
        mid = (left + right) // 2
        _div(nums, left, mid)
        _div(nums, mid + 1, right)

        # 归并
        _merge(nums, left, mid, right)

    def _merge(nums, left, mid, right):
        left_nums = copy.deepcopy(nums[left:mid + 1])
        right_nums = copy.deepcopy(nums[mid + 1:right + 1])
        len_left = mid - left + 1
        len_right = right - mid

        l_i = 0
        r_i = 0
        raw_i = left
        while l_i < len_left and r_i < len_right:
            if left_nums[l_i] <= right_nums[r_i]:
                nums[raw_i] = left_nums[l_i]
                l_i += 1
            else:
                nums[raw_i] = right_nums[r_i]
                r_i += 1
            raw_i += 1

        while l_i < len_left:
            nums[raw_i] = left_nums[l_i]
            l_i += 1
            raw_i += 1

        while r_i < len_right:
            nums[raw_i] = right_nums[r_i]
            r_i += 1
            raw_i += 1

    _div(nums, 0, len(nums) - 1)


def _quick_sort(nums):
    def _partition(nums, left, right):
        pivot = nums[left]
        while left < right:
            # 从右向左找第一个小于pivot的
            while left < right and nums[right] >= pivot:
                right -= 1
            # 换到前面
            nums[left] = nums[right]
            # 从左想右找第一个大于pivot的
            while left < right and nums[left] <= pivot:
                left += 1
            # 换到后面
            nums[right] = nums[left]

        # 替换left位置
        nums[left] = pivot
        return left

    def _ite(nums, left, right):
        if left >= right:
            return
        mid = _partition(nums, left, right)
        _ite(nums, left, mid - 1)
        _ite(nums, mid + 1, right)

    _ite(nums, 0, len(nums) - 1)


if __name__ == '__main__':
    nums = [5, 8, 1, 7, 3, 2, 5]
    _quick_sort(nums)
    print(nums)
