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

from tensorflow.keras.applications.resnet50 import ResNet50

resnet50_base_model = ResNet50(weights='imagenet', include_top=False)

#resnet50_base_model.summary()

## building base model with frozen weights
base_x = resnet50_base_model.output
global_pooling_x = GlobalAveragePooling2D()(base_x)
#add dense layer
dense_layer = Dense(512, activation='relu')(global_pooling_x)
#add final output layer
base_prediction = Dense(3, activation = 'sigmoid')(dense_layer)

#create the whole network
resnet_transfer_model_1 = Model(inputs=resnet50_base_model.input, outputs=base_prediction)


#resnet_transfer_model_1.summary()


## set train, validation, and test paths
train_path = '//Desktop/sudhir/GroceryStoreDataset-master/dataset/train' 
validation_path = '/Desktop/sudhir/GroceryStoreDataset-master/dataset/val'
test_path = '/Desktop/sudhir/GroceryStoreDataset-master/dataset/test'



#preprocessing the data

train_batches  = ImageDataGenerator().flow_from_directory(
    train_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 30)

validation_batches  = ImageDataGenerator().flow_from_directory(
    validation_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 10)

test_batches  = ImageDataGenerator().flow_from_directory(
    test_path, target_size=(224,224), classes = ['Fruits', 'Packages', 'Vegetables'], batch_size = 10)

train_filenames = train_batches.filenames
steps_train = len(train_filenames)/train_batches.batch_size
#print(steps_train)

## set steps per epoch for validation
validation_filenames = validation_batches.filenames
steps_valid = len(validation_filenames)/validation_batches.batch_size
#print(steps_valid)




resnet_transfer_model_1.compile(loss='categorical_crossentropy',
              optimizer= SGD(learning_rate = 1e-5 ),
              metrics=['categorical_accuracy'])
resnet50_base_model.save('/Desktop/sudhir/ResNet50.h5', include_optimizer=True, save_traces=True)

# resnet_model_1_fit_generator = resnet_transfer_model_1.fit(
#         train_batches,
#         steps_per_epoch=steps_train,
#         epochs=5,
#         validation_data = validation_batches,
#         validation_steps=steps_valid)


# plt.plot(resnet_model_1_fit_generator.history['loss'])
# plt.plot(resnet_model_1_fit_generator.history['val_loss'])
# plt.title('ResNet50 loss-SGD')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()


# plt.plot(resnet_model_1_fit_generator.history['categorical_accuracy'])
# plt.plot(resnet_model_1_fit_generator.history['val_categorical_accuracy'])
# plt.title('ResNet50  accuracy-SGD')
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
#     prediction = resnet_transfer_model_1.predict(img_preprocessed)

#     print(prediction)

# classify("/Desktop/sudhir/GroceryStoreDataset-master/sample_images/iconic/Lemon_Iconic.jpg")
