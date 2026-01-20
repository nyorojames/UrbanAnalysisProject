import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("tea.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

#rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([36,25,25])
upper_green = np.array([70,255,255])

lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 60, 255])

green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
grey_mask = cv2.inRange(hsv_img, lower_gray, upper_gray)

result = cv2.bitwise_and(img, img, mask=green_mask)

green_pixels = np.count_nonzero(green_mask)
grey_pixels = np.count_nonzero(grey_mask)

height, width = img.shape[:2]
total_pixels = height * width

green_ratio = (green_pixels / total_pixels) * 100
grey_ratio = (grey_pixels / total_pixels) * 100

text = f"Green area: {green_ratio:.2f}%"
text2 = f"Built area: {grey_ratio:.2f}%"

cv2.putText(result, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
cv2.putText(result, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

# plt.figure()
# plt.imshow(rgb_img)
# plt.show()
#
cv2.imshow("Original", img)
# cv2.imshow("Green Mask", green_mask)
cv2.imshow("Grey Mask", grey_mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()