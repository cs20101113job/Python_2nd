import cv2
img_color = cv2.imread("./images/rgb.png")
cv2.imshow("Color (BGR)", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()