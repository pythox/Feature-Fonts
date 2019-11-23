import pytesseract
import cv2
import numpy as np
from pytesseract import Output
img = cv2.imread('test.jpg')

d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
n_boxes = len(d['level'])

words = []

for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    words.append(img[y:y+h, x:x+w])

print(np.shape(words))

cv2.waitKey(0)
