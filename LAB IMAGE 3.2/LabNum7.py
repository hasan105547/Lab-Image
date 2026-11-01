
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'F:\LabAccess\LAB IMAGE 3.2\Lab7.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (11, 11), 0)

_, binary = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if np.mean(binary[0, :]) > 127:
    binary = cv2.bitwise_not(binary)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((5, 5), np.uint8)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
count = 0

for c in contours:
    area = cv2.contourArea(c)

    if area > 1500:
        count += 1
        x, y, w, h = cv2.boundingRect(c)

        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

        cv2.putText(result, str(count), (x+w//2-10, y+h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(result, f'A: {int(area)}', (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        print(f'Coin {count} Area: {area:.2f} pixels')

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(clean, cmap='gray')
plt.title('Binary Image (Cleaned)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'Labeled Coins with Area (Total: {count})')
plt.axis('off')

plt.tight_layout()
plt.show()

print('Total Coins counted successfully:', count)
