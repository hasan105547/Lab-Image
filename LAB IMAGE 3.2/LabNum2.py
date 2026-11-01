# Lab 2
# Write a MATLAB/Python program to read ‘rice.tif’ image, count number of 
# rice and display area (also specific range), major axis length, and perimeter. 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import label, regionprops

# 1. Read image
image = cv2.imread(r"rice (1).tif ", 0)


# 2. Convert image to binary
_, binary = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3. Label separate rice grains
labeled_image = label(binary > 0)
regions = regionprops(labeled_image)

# Remove very small objects (noise)
regions = [r for r in regions if r.area >= 20]

# 4. Count rice grains
print("Total number of rice:", len(regions))

# 5. Display area, major axis length and perimeter
data = []

for i, r in enumerate(regions, start=1):
    data.append([
        i,
        r.area,
        round(r.major_axis_length, 2),
        round(r.perimeter, 2)
    ])

df = pd.DataFrame(data, columns=[
    "Rice No", "Area", "Major Axis Length", "Perimeter"
])

print("\nMeasurements of Rice Grains:")
print(df.to_string(index=False))

# 6. Display rice grains within a specific area range
min_area = 100
max_area = 200

selected = df[
    (df["Area"] >= min_area) &
    (df["Area"] <= max_area)
]

print("\nRice grains with area between", min_area, "and", max_area)
print(selected.to_string(index=False))
print("Number of selected rice grains:", len(selected))

# 7. Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(labeled_image, cmap="nipy_spectral")
plt.title("Labeled Rice Grains")
plt.axis("off")

plt.tight_layout()
plt.show()