import cv2
import numpy as np

# Read and convert image to grayscale
image = cv2.imread("lena5.jpg")
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray_img)

print("\nOriginal grayscale matrix:\n", gray_img)

# Get image dimensions
rows, cols = gray_img.shape
total_pixels = rows * cols

# Flatten image to 1D
flat_pixels = gray_img.flatten()

# Sort and get min/max
sorted_pixels = np.sort(flat_pixels)
min_pixel = sorted_pixels[0]
max_pixel = sorted_pixels[-1]

# Get unique pixel values
unique_vals = np.unique(flat_pixels)

# Count pixel frequencies
freq = [np.count_nonzero(flat_pixels == val) for val in unique_vals]

# Compute cumulative distribution function (CDF)
cdf = np.cumsum(freq)

# Map old pixel values to new equalized values
cdf_min = cdf[0]
equalized_vals = np.round(((cdf - cdf_min) / (total_pixels - cdf_min)) * 255).astype(np.uint8)

# Create mapping dictionary
pixel_map = dict(zip(unique_vals, equalized_vals))

# Apply mapping to get equalized image
equalized_img = np.vectorize(pixel_map.get)(gray_img).astype(np.uint8)

print("\nEqualized image matrix:\n", equalized_img)

# Show results
cv2.imshow("Histogram Equalized", equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
