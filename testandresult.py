from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
import keras

PATHFILE = "contest"
FILENAME = 'filelist.txt'
IM_SIZE = 128
BATCH_SIZE = 50

datagen = ImageDataGenerator(rescale=1/255.)

# modelName = 'model1.h5' #acc = 0.7625
# modelName = 'model1v5.h5' #acc = 0.7699

# modelName = 'model2.h5' #acc = 0.7649
# modelName = 'model2v2.h5' #acc = 0.7624

# modelName = 'model3.h5' #acc = 0.7749
# modelName = 'model3v2.h5' #acc = 0.7674
modelName = 'model3v3.h5' #acc = 0.8025

# modelName = 'model4.h5' #acc = 0.7099
# modelName = 'model4v2.h5' #acc = 0.7275

filename = os.listdir(PATHFILE)
fclass = {0: 'B', 1:'D', 2:'R', 3:'S'}
model = load_model(modelName)

file = open("result2.txt","w")
for name in filename:
    test_im = cv2.imread(PATHFILE + "/" + name, cv2.IMREAD_COLOR)
    test_im = cv2.resize(test_im, (IM_SIZE,IM_SIZE))
    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = test_im / 255.
    test_im = np.expand_dims(test_im, axis=0)
    w_pred = model.predict(test_im)
    
    argMax = np.argmax(w_pred,axis = -1)[0]
    file.write(name + "::" + fclass[argMax] + "\n")
    # print()
file.close()
