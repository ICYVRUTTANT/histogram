import cv2
import matplotlib.pyplot as plt
import os

folder_path = r"C:\Users\vrutt\Desktop\vruttant\dip\HISTOGRAM"
image_name = "TOTA"

# Automatically detect image with common extensions
def find_image(base_name, folder="."):
    for file in os.listdir(folder):
        if file.startswith(base_name) and file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            return os.path.join(folder, file)
    return None

image_path = find_image(image_name, folder_path)
if image_path is None:
    raise FileNotFoundError(f"Image '{image_name}' not found in folder '{folder_path}'")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Failed to read the image.")

equalized_img = cv2.equalizeHist(img)
resized_img = cv2.resize(equalized_img, (800, 600))

plt.imshow(resized_img, cmap='gray')
plt.title("Histogram Equalized Image")
plt.axis('off')
plt.show()
