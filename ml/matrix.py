## u have a 2d array representing grayscales pixel values (8*8 matrix). Reshape it into a 4D array where each 4*4 subarray represents a quadrant of the image

import numpy as np

# 8x8 matrix
pixel_matrix = np.arange(64).reshape(8, 8)

# Each quadrant becomes a separate 4x4 array
reshaped_array = np.zeros((2, 2, 4, 4))

# Upper left quadrant
reshaped_array[0, 0] = pixel_matrix[0:4, 0:4]

# Upper right quadrant
reshaped_array[0, 1] = pixel_matrix[0:4, 4:8]

# Lower left quadrant
reshaped_array[1, 0] = pixel_matrix[4:8, 0:4]

# Lower right quadrant
reshaped_array[1, 1] = pixel_matrix[4:8, 4:8]

##The resulting reshaped_array is a 4D array with shape (2, 2, 4, 4) where:

# The first two dimensions (2, 2) represent the row and column position of each quadrant
# The last two dimensions (4, 4) represent the pixel values within each quadrant
