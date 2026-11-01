
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'C:\Users\LENOVO\Downloads\lab1.jpg'

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, [0, 256])
plt.title('Histogram')

kernel_lp = np.ones((5, 5), np.float32) / 25
low_pass_img = cv2.filter2D(img, -1, kernel_lp)

plt.subplot(2, 2, 3)
plt.imshow(low_pass_img, cmap='gray')
plt.title('Low Pass Filtered Image')

kernel_hp = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]])

high_pass_img = cv2.filter2D(img, -1, kernel_hp)

plt.subplot(2, 2, 4)
plt.imshow(high_pass_img, cmap='gray')
plt.title('High Pass Filtered Image')

plt.tight_layout()
plt.show()
