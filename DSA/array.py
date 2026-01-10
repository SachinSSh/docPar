# Creating a list
arr = [1, 2, 3, 4, 5]

# Accessing elements (0-indexed)
first_element = arr[0]  # 1
last_element = arr[-1]  # 5

# Slicing
sub_array = arr[1:4]  # [2, 3, 4]

# Modifying elements
arr[0] = 10  # [10, 2, 3, 4, 5]

# Length
length = len(arr)  # 5

# Adding elements
arr.append(6)  # [10, 2, 3, 4, 5, 6]
arr.insert(1, 15)  # [10, 15, 2, 3, 4, 5, 6]
arr.extend([7, 8])  # [10, 15, 2, 3, 4, 5, 6, 7, 8]

# Removing elements
arr.pop()  # Removes and returns the last element
arr.pop(1)  # Removes element at index 1
arr.remove(3)  # Removes the first occurrence of value 3

# Checking if an element exists
is_present = 4 in arr  # True/False

# Finding index of an element
index = arr.index(4)  # Returns index of first occurrence of 4
