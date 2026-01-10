def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1  # Element not found
