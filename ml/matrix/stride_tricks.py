from numpy.lib import stride_tricks

# Create sliding windows using stride tricks
arr = np.arange(10)
window_size = 3
stride = 1

# Calculate new shape and strides for the view
shape = ((arr.size - window_size) // stride + 1, window_size)
strides = (arr.itemsize * stride, arr.itemsize)

# Create sliding windows
windows = stride_tricks.as_strided(arr, shape=shape, strides=strides)
print("Sliding windows:")
print(windows)

# 2D convolution-like window views
arr_2d = np.arange(36).reshape(6, 6)
window_shape = (3, 3)
windows_2d = np.lib.stride_tricks.sliding_window_view(arr_2d, window_shape)
print("\n2D sliding window view shape:", windows_2d.shape)
print("First 3x3 window:")
print(windows_2d[0, 0])
