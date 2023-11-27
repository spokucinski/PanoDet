import cv2

global mouseX,mouseY

def draw_circle(event,x,y,flags,param):   
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y

def draw_line(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        i = 4
        
img = cv2.imread("C:\\Users\\Admin\\Downloads\\Bedroom.jpg")

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print (mouseX, mouseY)