# Left rotation by k positions
def left_rotate(arr, k):
    k = k % len(arr)
    return arr[k:] + arr[:k]

# Right rotation by k positions
def right_rotate(arr, k):
    k = k % len(arr)
    return arr[-k:] + arr[:-k]
