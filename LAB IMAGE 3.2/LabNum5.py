import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read Image
img = cv2.imread(r'C:\Users\LENOVO\Downloads\LabNum5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Preprocessing
blur = cv2.GaussianBlur(gray, (3, 3), 0)
_, binary = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((2, 2), np.uint8)
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 3. Find and Sort Characters
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

chars = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if cv2.contourArea(c) > 20 and h > 10:
        chars.append((x, y, w, h))

chars.sort(key=lambda c: c[0])

# 4. Segment Characters
result = img.copy()
segments = []

for i, (x, y, w, h) in enumerate(chars):
    char = gray[max(0,y-3):y+h+3, max(0,x-3):x+w+3]
    segments.append(char)

    cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(result, str(i+1), (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# 5. Display Results
plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate([
    (img, 'Original Image'),
    (binary, 'Binary Image'),
    (result, 'Character Segmentation')
]):
    plt.subplot(2, 2, i+1)
    plt.imshow(image if i == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               if i == 0 or i == 2 else image,
               cmap='gray' if i == 1 else None)
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# 6. Show Individual Characters
plt.figure(figsize=(14, 4))

for i, char in enumerate(segments):
    plt.subplot(1, len(segments), i+1)
    plt.imshow(char, cmap='gray')
    plt.title(str(i+1))
    plt.axis('off')

plt.show()

print("Total Characters:", len(segments))


