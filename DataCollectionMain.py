import WebcamModule as wM
import DataCollectionModule as dcM
import JoyStickModule as jsM
import cv2
from time import sleep


record = 0
while True:
    joyVal = jsM.getJS()
    print(joyVal)
    steering = joyVal['axis1']
    if joyVal['share'] == 1:
        if record ==0: print('Recording Started ...')
        record +=1
        sleep(0.300)
    if record == 1:
        img = wM.getImg(True,size=[240,120])
        dcM.DataKaydet(img,steering)
    elif record == 2:
        dcM.saveLog()
        record = 0

