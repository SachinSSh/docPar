# Using built-in functions
maximum = max(arr)
minimum = min(arr)

# Manual implementation
def find_max(arr):
    max_val = float('-inf')
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val
