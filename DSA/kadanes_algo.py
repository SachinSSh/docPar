def kadane(arr):
    # Finds maximum sum subarray
    max_so_far = max_ending_here = arr[0]
    
    for num in arr[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
        
    return max_so_far
