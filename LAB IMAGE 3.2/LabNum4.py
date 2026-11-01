import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = r'C:\Users\LENOVO\Downloads\lab1.jpg'

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

laplacian_cv = cv2.Laplacian(img, cv2.CV_64F)
laplacian_cv_abs = cv2.convertScaleAbs(laplacian_cv)

kernel_laplacian = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=np.float32)

laplacian_custom = cv2.filter2D(img, -1, kernel_laplacian)

img_sharpened = cv2.addWeighted(img, 1.0, laplacian_custom, 0.7, 0)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Original Image')

plt.subplot(2, 2, 2)
plt.imshow(laplacian_cv_abs, cmap='gray')
plt.title('2. OpenCV cv2.Laplacian()')

plt.subplot(2, 2, 3)
plt.imshow(laplacian_custom, cmap='gray')
plt.title('3. Custom 3x3 Mask Kernel')

plt.subplot(2, 2, 4)
plt.imshow(img_sharpened, cmap='gray')
plt.title('4. Sharpened (Original + Laplacian)')

plt.tight_layout(pad=2.0)
plt.show()