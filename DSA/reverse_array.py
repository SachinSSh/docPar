# Using slicing
reversed_arr = arr[::-1]

# In-place reversal
def reverse_array(arr, start, end):
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1
    return arr
