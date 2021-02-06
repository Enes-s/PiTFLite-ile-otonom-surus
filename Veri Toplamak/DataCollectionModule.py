import pandas as pd
import os 
import cv2
from datetime import datetime

global imgList,steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

BenimYolum = os.path.join(os.getcwd(),'DataCollected')
#print(BenimYolum)

#Yeni Dosya oluşumu
while os.path.exists(os.path.join(BenimYolum,f'IMG{str(countFolder)}')):
        countFolder += 1 # her seferinde dosyayı 1er arttırıp yeni dosya oluşturur

YeniYol = BenimYolum +"/IMG"+str(countFolder)
os.makedirs(YeniYol)

#IMG Kaydetme
def DataKaydet(img,steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    #print("timestamp =", timestamp)
    fileName = os.path.join(YeniYol,f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName, img)
    imgList.append(fileName)
    steeringList.append(steering)

# SAVE LOG FILE WHEN THE SESSION ENDS
def saveLog(): # csv dosyası şeklinde kayıt tutma
    global imgList, steeringList
    rawData = {'Image': imgList,
                'Steering': steeringList} # img leri image listine steeringleri steering listine
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(BenimYolum,f'log_{str(countFolder)}.csv'), index=False, header=False) # csv formatında kaydet
    print('Log Saved')
    print('Total Images: ',len(imgList)) # toplam fotoğraf sayısı
    
  
if __name__ == '__main__': #main bloğunda çalıştır
        cap = cv2.VideoCapture(0)# 1. kayıt cihazını açar
        for x in range(10): # belirlenen x sayısı kadar döngü yaratır
            _, img = cap.read()
            DataKaydet(img, 0.5) # steering açısını kendimiz veriyoruz!
            cv2.waitKey(0)
            cv2.imshow("Image", img)
        saveLog()
