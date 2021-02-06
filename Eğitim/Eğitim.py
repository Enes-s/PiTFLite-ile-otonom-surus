print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.append('PATH_TO_TENSORFLOW_OBJECT_DETECTION_FOLDER')
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import tf


1.Aşama Datayı Eğitim aşamasına alma
path = 'DataCollected'
data = importDataInfo(path)
print(data.head())

2.Aşama Datadaki dengeyi sağlamak ve Görselleştirmek
data = balanceData(data,display=True)

3.Aşama Preprocess için hazırlamak
imagesPath, steerings = loadData(path,data)
# print('No of Path Created for Images ',len(imagesPath),len(steerings))
# cv2.imshow('Test Image',cv2.imread(imagesPath[5]))
# cv2.waitKey(0)

4.Aşama Train test split
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                              test_size=0.3,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

5. Aşama veriyi düzenleme

6.Aşama Preprocess

7.Aşama Modeli oluşturma
model = createModel()

EarlyStopping = EarlyStopping(monitor = "val_loss",mode = "min",verbose = 1,patience = 50)
model.fit(x=xTrain,y=yTrain,epochs = 300,validation_data = (xVal,yVal),verbose = 1,callbacks=[EarlyStopping])
model.save('model3.h5')
print('Model Saved')
#%%
8.Aşama Eğitim
history = model.fit(dataGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=50,
                                  epochs=15,
                                  validation_data=dataGen(xVal, yVal, 50, 0),
                                  validation_steps=50)

9.Aşama Modeli Kayıt etme
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
print('Model Saved')
#%%
10.Aşama Sonuç Modelini Görselleştirme
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
#%%
loss = history.history["loss"]
sns.lineplot(x=range(len(loss)), y =loss)
