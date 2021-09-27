import numpy as np
import cv2 as cv
import HCD
import time

frame_rate = 10
prev = 0

cap = cv.VideoCapture('20210925_203755.mp4')

##frame_w = int(cap.get(3))
##frame_h = int(cap.get(4))

frame_width = int( cap.get(cv.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'MJPG',)

out = cv.VideoWriter('out.avi', fourcc, 10, (frame_width, frame_height))

##size = (frame_w,frame_h)
##frame_time = 100
##fourcc = cv.VideoWriter_fourcc('M','P','E','G')
##out = cv.VideoWriter('bench.avi',fourcc,frame_rate,(frame_w,frame_h))
##out = cv.VideoWriter('bench.avi',-1,frame_rate,(frame_w,frame_h))

if (cap.isOpened() == False): 
    print("Error reading video file")
    
while(cap.isOpened()):
    time_elapsed = time.time() - prev
    res, frame = cap.read()
    frame = cv.resize(frame, (frame_width,frame_height))
    print(time.time())
    if time_elapsed > 1./frame_rate:
        corner_list,corner,R = HCD.harris_corner(frame)
        t = len(corner_list)
#       corner = cv.resize(corner, (640,480))
        cv.putText(corner, str(t), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_4)
        cv.imshow('Frame',corner)
        out.write(corner)
        prev = time.time()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


       
cap.release()
out.release
cv.destroyAllWindows()



