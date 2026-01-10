def sliding_window_fixed(arr, k):
    # k is the window size
    n = len(arr)
    result = []
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    result.append(window_sum)
    
    # Slide window and calculate sums
    for i in range(n - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        result.append(window_sum)
        
    return result

def sliding_window_variable(arr, target_sum):
    # Find smallest subarray with sum >= target_sum
    n = len(arr)
    current_sum = 0
    min_length = float('inf')
    start = 0
    
    for end in range(n):
        current_sum += arr[end]
        
        while current_sum >= target_sum:
            min_length = min(min_length, end - start + 1)
            current_sum -= arr[start]
            start += 1
            
    return min_length if min_length != float('inf') else 0
