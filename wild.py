#wildfire detection with CNN
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2

train_directory='/Users/bluh/Downloads/wildfire/train'
test_directory='/Users/bluh/Downloads/wildfire/test'
val_directory='/Users/bluh/Downloads/wildfire/valid'

training_images = []
training_labels = []
folderdict={'nowildfire':0,'wildfire':1}
for folder in os.listdir(train_directory):
    folder_path = os.path.join(train_directory, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(train_directory+'/'+folder):
            img=cv2.imread(train_directory+'/'+folder+'/'+file)
            img=cv2.resize(img, (32,32))
            img=np.array(img)
            img=img/255.0
            training_images.append(img)
            training_labels.append(folderdict[folder])

test_images = []
test_labels = []
for folder in os.listdir(test_directory):
    folderpath = os.path.join(test_directory, folder)
    if os.path.isdir(folderpath):
        for file in os.listdir(test_directory+'/'+folder):
            img=cv2.imread(test_directory+'/'+folder+'/'+file)
            img=cv2.resize(img, (32,32))
            img=np.array(img)
            img=img/255.0
            test_images.append(img)
            test_labels.append(folderdict[folder])

val_images = []
val_labels = []
for folder in os.listdir(val_directory):
    folderpth = os.path.join(val_directory, folder)
    if os.path.isdir(folderpth):
        for file in os.listdir(val_directory+'/'+folder):
            img=cv2.imread(val_directory+'/'+folder+'/'+file)
            img=cv2.resize(img, (32,32))
            img=np.array(img)
            img=img/255.0
            val_images.append(img)
            val_labels.append(folderdict[folder])


training_images=np.array(training_images)
training_labels=np.array(training_labels)

test_images=np.array(test_images)
test_labels=np.array(test_labels)

val_images=np.array(val_images)
val_labels=np.array(val_labels)



from keras import Sequential
from keras import datasets, layers, models

model = models.Sequential()
model.add(layers.InputLayer(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        # 32 output filters
        # (3, 3) kernel size
        # input shape: 350x350x3 (3 - images colored)
model.add(layers.MaxPooling2D((2, 2)))
        # 2x2 pooling window size
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class PrintMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}, val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}")

history = model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_acc*100))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()