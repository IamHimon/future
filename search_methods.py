# 查找算法
# 1. 二分查找

def binary_search(nums, target):
    """
    递归实现
    :param nums:
    :param target:
    :return:
    """
    def _search(left, right, nums, target):
        if left > right:
            return -1
        mid = (left + right) // 2
        mid_value = nums[mid]
        if target == mid_value:
            return mid
        if target < mid_value:
            return _search(left, mid - 1, nums, target)
        if target > mid_value:
            return _search(mid + 1, right, nums, target)

    index = _search(0, len(nums) - 1, nums, target)
    return index


def binary_search2(nums, target):
    """
    非递归实现
    :param nums:
    :param target:
    :return:
    """
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        mid_value = nums[mid]
        if target < mid_value:
            right = mid - 1
        elif target > mid_value:
            left = mid + 1
        else:
            return mid
    return -1


if __name__ == '__main__':
    nums = [1, 2, 3, 4, 5, 6, 7, 8]
    res = binary_search2(nums, 4)
    print(res)
