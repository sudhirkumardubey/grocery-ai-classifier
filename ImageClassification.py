import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import csv
import scipy
import matplotlib.pyplot as plt


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.preprocessing import image
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten
from keras import backend as K


## set train, validation, and test paths
train_path = '/Users/dikshabhandari/Desktop/sudhir/GroceryStoreDataset-master/dataset/train' 
validation_path = '/Users/dikshabhandari/Desktop/sudhir/GroceryStoreDataset-master/dataset/val'
test_path = '/Users/dikshabhandari/Desktop/sudhir/GroceryStoreDataset-master/dataset/test'



#preprocessing the data

train_batches  = ImageDataGenerator().flow_from_directory(
    train_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 30)

validation_batches  = ImageDataGenerator().flow_from_directory(
    validation_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 10)

test_batches  = ImageDataGenerator().flow_from_directory(
    test_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 10)


#creating Convolutional neural network
from keras import layers
from keras import models


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='sigmoid'))
#model.summary()


#compile and run

## set steps per epoch for train
train_filenames = train_batches.filenames
steps_train = len(train_filenames)/train_batches.batch_size
#print(steps_train)

## set steps per epoch for validation
validation_filenames = validation_batches.filenames
steps_valid = len(validation_filenames)/validation_batches.batch_size
#print(steps_valid)

model.compile(loss='categorical_crossentropy',
              optimizer= SGD(learning_rate = 1e-4), metrics = 'categorical_accuracy')

model.save('/Users/dikshabhandari/Desktop/sudhir/baseline_4layer.h5', include_optimizer=True, save_traces=True)

# fit_generator1 = model.fit(
#       train_batches,
#       steps_per_epoch=steps_train,
#       epochs=20,
#       validation_data = validation_batches,
#       validation_steps=steps_valid)

# plt.plot(fit_generator1.history['loss'])
# plt.plot(fit_generator1.history['val_loss'])
# plt.title('Model loss-SGD')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()


# plt.plot(fit_generator1.history['categorical_accuracy'])
# plt.plot(fit_generator1.history['val_categorical_accuracy'])
# plt.title('Model accuracy-SGD')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()



# def classify(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)

#     img_batch = np.expand_dims(img_array, axis=0)

#     img_preprocessed = preprocess_input(img_batch)

#     #model = tf.keras.applications.resnet50.ResNet50()
#     prediction = model.predict(img_preprocessed)

#     print(prediction)

# classify("/Users/dikshabhandari/Desktop/sudhir/GroceryStoreDataset-master/sample_images/iconic/Alpro-Fresh-Soy-Milk_Iconic.jpg")
