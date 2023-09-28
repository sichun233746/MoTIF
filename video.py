import cv2
import os
import random


#img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/000001.png")
img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/merge/0.png")
fps = 20
size = (img.shape[1], img.shape[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
videoWriter = cv2.VideoWriter('./test/merge_20.avi', fourcc, fps, size)

for i in range(0,1096):
    #img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/"+'{:06d}'.format(i+1)+".png")
    img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/merge/"+str(i)+".png")
    videoWriter.write(img)
videoWriter.release()

#img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/000001.png")
#img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/Ours/0.png")
img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/output/1114/GOPR0384_11_00/Ours/0.png")
fps = 20
size = (img.shape[1], img.shape[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'VP90')
videoWriter = cv2.VideoWriter('./test/Ours_GOPR0384_11_00.webm', fourcc, fps, size)
for i in range(0,481):
    if i > 482:
        break
    print(i)
    #img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/"+'{:06d}'.format(i+1)+".png")
    #img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/Ours/"+str(i)+".png")
    img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/output/1114/GOPR0384_11_00/Ours/"+str(i)+".png")
    videoWriter.write(img)
videoWriter.release()

#img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/000001.png")
#img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/INR/0.png")
img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/output/1114/GOPR0384_11_00/VideoINR_32/0.png")
fps = 20
size = (img.shape[1], img.shape[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'VP90')
videoWriter = cv2.VideoWriter('./test/INR_GOPR0384_11_00_32.webm', fourcc, fps, size)
for i in range(0,120):
    if i > 482:
        break
    print(i)
    #img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/"+'{:06d}'.format(i+1)+".png")
    #img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/test/GOPR0871_11_00/INR/"+str(i)+".png")
    img = cv2.imread("/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/output/1114/GOPR0384_11_00/VideoINR_32/"+str(i)+".png")
    videoWriter.write(img)
videoWriter.release()

img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/000001.png")
fps = 20
size = (img.shape[1], img.shape[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'VP90')
videoWriter = cv2.VideoWriter('./test/GT.webm', fourcc, fps, size)
for i in range(0,1096):
    if i > 500:
        break
    print(i)
    img = cv2.imread("/work/abcd233746pc/test/GOPR0871_11_00/"+'{:06d}'.format(i+1)+".png")
    videoWriter.write(img)
videoWriter.release()

img = cv2.imread("/work/abcd233746pc/test_4/GOPR0871_11_00/000001.png")
img = cv2.resize(img, (320*4,180*4),interpolation=cv2.INTER_NEAREST)
fps = 20/8
size = (img.shape[1], img.shape[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'VP90')
videoWriter = cv2.VideoWriter('./test/Input_up.webm', fourcc, fps, size)
for i in range(0,1096,8):
    if i > 500:
        break
    print(i)
    img = cv2.imread("/work/abcd233746pc/test_4/GOPR0871_11_00/"+'{:06d}'.format(i+1)+".png")
    img = cv2.resize(img, (320*4,180*4),interpolation=cv2.INTER_NEAREST)
    videoWriter.write(img)
videoWriter.release()


