# coding=utf-8
import cv2

d = 0
color = 0
space = 0

def change_d(x):
    blur = cv2.bilateralFilter(img, x, color, space)
    cv2.imshow("myImg", blur)

def change_color(x):
    blur = cv2.bilateralFilter(img, d, x, space)
    cv2.imshow("myImg", blur)


def change_space(x):
    blur = cv2.bilateralFilter(img, d, color, x)
    cv2.imshow("myImg", blur)


img = cv2.imread('JD/bird.jpg')

cv2.namedWindow('myImg')
cv2.createTrackbar('d', 'myImg', 1, 500, change_d)
cv2.createTrackbar('color', 'myImg', 1, 500, change_color)
cv2.createTrackbar('space', 'myImg', 1, 500, change_space)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    cv2.getTrackbarPos('d', 'myImg')
    cv2.getTrackbarPos('color', 'myImg')
    cv2.getTrackbarPos('space', 'myImg')

cv2.destroyAllWindows()



