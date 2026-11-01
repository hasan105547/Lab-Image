
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Lab7.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

enhance = cv2.equalizeHist(gray)
avg = cv2.blur(gray, (5, 5))

gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

median = cv2.medianBlur(gray, 5)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharp = cv2.filter2D(gray, -1, kernel)
images = [
    (gray, 'Original Image'),
    (enhance, 'Histogram Equalization'),
    (avg, 'Average Smoothing'),
    (gaussian, 'Gaussian Smoothing'),
    (median, 'Median Smoothing'),
    (sharp, 'Sharpened Image')
]
plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(images):
    plt.subplot(2, 3, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
