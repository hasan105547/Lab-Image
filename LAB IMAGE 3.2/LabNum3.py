import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r'C:\Users\LENOVO\Downloads\lab1.jpg'

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

kernel_blur = np.ones((3, 3), np.float32) / 9.0

kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

kernel_edge = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]], dtype=np.float32)

img_blur = cv2.filter2D(img, -1, kernel_blur)
img_sharpen = cv2.filter2D(img, -1, kernel_sharpen)
img_edge = cv2.filter2D(img, -1, kernel_edge)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Original Image')

plt.subplot(2, 2, 2)
plt.imshow(img_blur, cmap='gray')
plt.title('2. Blurred Image (3x3 Averaging)')

plt.subplot(2, 2, 3)
plt.imshow(img_sharpen, cmap='gray')
plt.title('3. Sharpened Image (3x3 Sharpen)')

plt.subplot(2, 2, 4)
plt.imshow(img_edge, cmap='gray')
plt.title('4. Edge Detected Image (3x3 Edge)')

plt.tight_layout(pad=2.0)
plt.show()