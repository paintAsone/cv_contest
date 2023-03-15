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

#acc = 0.762 OK
model1 = Sequential([
    keras.layers.Conv2D(8, 3, activation='relu', input_shape = (IM_SIZE,IM_SIZE,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(16, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation ='softmax')
])

# acc = 0.7629 OK
model2 = Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape = (IM_SIZE,IM_SIZE,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation ='softmax')
])

#acc = 0.685 saved in model3v2.h5
model3 = Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape = (IM_SIZE,IM_SIZE,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    # keras.layers.Conv2D(64, 3, activation='relu'),
    # keras.layers.MaxPooling2D(),
    # keras.layers.Conv2D(128, 3, activation='relu'),
    # keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation ='softmax')
])

model4 = Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape = (IM_SIZE,IM_SIZE,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    # keras.layers.Conv2D(128, 3, activation='relu'),
    # keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation ='softmax')
])

model4.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model4.summary()

#Create generator
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'data/train',
    shuffle=True,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode = 'rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    'data/valid',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'data/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical')

# modelName = 'model1.h5' 
# modelName = 'model1v2.h5' 

# modelName = 'model2.h5'
# modelName = 'modelv2.h5'

# modelName = 'model3.h5'
# modelName = 'model3v2.h5'

# modelName = 'model4.h5'
modelName = 'model4v2.h5'

#Train Model
checkpoint = ModelCheckpoint(modelName, verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model4.fit_generator(
    train_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])

#Test Model
# model = load_model(modelName)
# score = model.evaluate_generator(
#     test_generator,
#     steps=len(test_generator))
# print('score (cross_entropy, accuracy):\n',score)


# # test_generator.reset()
# predict = model.predict_generator(
#     test_generator,
#     steps=len(test_generator),
#     workers = 1,
#     use_multiprocessing=False)
# print('confidence:\n', predict)

# predict_class_idx = np.argmax(predict,axis = -1)
# print('predicted class index:\n', predict_class_idx)

# mapping = dict((v,k) for k,v in test_generator.class_indices.items())
# predict_class_name = [mapping[x] for x in predict_class_idx]
# print('predicted class name:\n', predict_class_name)

# cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
# print("Confusion Matrix:\n",cm)

# fclass = ['B','D','R','S']

# file = open("result.txt","w")
# n = 0
# for i in fclass:
#     path = './data/test/'+str(i)
#     # print(path)
#     filename = os.listdir(path)    
#     for f in filename:
#         file.write(f+":"+predict_class_name[n])
#         file.write('\n')
#         n+=1

# file.close()

plt.show()
