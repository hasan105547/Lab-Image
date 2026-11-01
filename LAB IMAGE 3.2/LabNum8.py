
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Lab1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 127, 255,
                             cv2.THRESH_BINARY)

gamma = 0.5
power = np.uint8(255 * (gray / 255) ** gamma)

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

highpass = cv2.filter2D(gray, -1, kernel)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(threshold, cmap='gray')
plt.title('Threshold Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(power, cmap='gray')
plt.title('Power Enhanced Image')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(highpass, cmap='gray')
plt.title('High Pass Image')
plt.axis('off')

plt.tight_layout()
plt.show()
