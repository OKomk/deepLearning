import cv2

file = cv2.imread(r'C:\Users\Kshat\Documents\StudyMaterial\cs671\assn3\Group_18\train\0\img_1.jpg')
print(file)
cv2.imshow('Image',file)
cv2.waitKey(0)