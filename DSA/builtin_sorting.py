arr = [5, 2, 8, 1, 9]
arr.sort()  # In-place sorting: [1, 2, 5, 8, 9]
sorted_arr = sorted(arr)  # Returns new sorted array

# Custom sorting
arr.sort(reverse=True)  # Descending order
arr.sort(key=lambda x: abs(x))  # Sort by absolute value
