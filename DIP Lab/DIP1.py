import cv2
image=cv2.imread('C:/Users/harsh/Downloads/tictactoe.png')
height,width=image.shape[:2]
mid_height=height//2
mid_width=width//2
up=image[:mid_height, :mid_width]
cv2.imshow('Up',up)
down=image[mid_height:,:mid_width]
cv2.imshow('Down',down)
right=image[:mid_height,mid_width:]
cv2.imshow('Right',right)
left=image[mid_height:,mid_width:]
cv2.imshow('Left',left)
cv2.imwrite('up.jpg',up)
cv2.imwrite('down.jpg',down)
cv2.imwrite('right.jpg',right)
cv2.imwrite('left.jpg',left)
