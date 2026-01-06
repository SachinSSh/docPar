a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Vertical stacking
v_stack = np.vstack((a, b))
print("Vertical stack:")
print(v_stack)

# Horizontal stacking
h_stack = np.hstack((a, b))
print("\nHorizontal stack:")
print(h_stack)

# Depth stacking (for 3D arrays)
depth_stack = np.dstack((a, b))
print("\nDepth stack shape:", depth_stack.shape)
print(depth_stack)

# General concatenation along axis
concat_0 = np.concatenate((a, b), axis=0)  # Same as vstack
concat_1 = np.concatenate((a, b), axis=1)  # Same as hstack
print("\nConcatenate axis=0:")
print(concat_0)

# Splitting arrays
arr = np.arange(16).reshape(4, 4)
print("\nOriginal array for splitting:")
print(arr)

# Split into 2 equal parts horizontally
h_split = np.hsplit(arr, 2)
print("\nHorizontal split (2 parts):")
print(h_split[0])
print(h_split[1])

# Split into specific sections using indices
split_indices = np.split(arr, [1, 3], axis=0)
print("\nSplit at indices [1, 3] on axis 0:")
for i, section in enumerate(split_indices):
    print(f"Section {i}:")
    print(section)
