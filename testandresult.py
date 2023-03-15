from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import keras

IM_SIZE = 128
BATCH_SIZE = 50

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    'data/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

# modelName = 'model1.h5' #acc = 0.7625
# modelName = 'model1v5.h5' #acc = 0.7699

# modelName = 'model2.h5' #acc = 0.7649
# modelName = 'model2v2.h5' #acc = 0.7624

# modelName = 'model3.h5' #acc = 0.0.7749
modelName = 'model3v2.h5' #acc = 0.7674

# modelName = 'model4.h5' #acc = 0.7099
# modelName = 'model4v2.h5' #acc = 0.7275

# Test Model
model = load_model(modelName)
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)


# test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('confidence:\n', predict)

#print acc
predict_class_idx = np.argmax(predict,axis = -1)
print('predicted class index:\n', predict_class_idx)

mapping = dict((v,k) for k,v in test_generator.class_indices.items())
predict_class_name = [mapping[x] for x in predict_class_idx]
print('predicted class name:\n', predict_class_name)

cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
print("Confusion Matrix:\n",cm)

fclass = ['B','D','R','S']
file = open("result.txt","w")
# n = 0
# for i in fclass:
#     path = './data/test/'+str(i)
#     # print(path)
#     filename = os.listdir(path)    
#     for f in filename:
#         file.write(f+":"+predict_class_name[n])
#         file.write('\n')
#         n+=1

# write filenames
filenames = test_generator.filenames
for i in range(len(filenames)):
    file.write(str(filenames[i])+":"+predict_class_name[i])
    file.write('\n')

file.close()