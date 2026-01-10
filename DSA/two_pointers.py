# Two pointers from ends
def two_pointer_from_ends(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process arr[left] and arr[right]
        left += 1
        right -= 1

# Two pointers in same direction
def two_pointer_same_direction(arr):
    slow = fast = 0
    while fast < len(arr):
        # Process using slow and fast pointers
        slow += 1
        fast += 2
