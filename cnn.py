from pickletools import optimize
from xml.etree.ElementInclude import include
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
from keras.applications.vgg16 import VGG16 
from keras.applications import vgg16
number_classes=2
# # Initialising the CNN
classifier = Sequential()

# # Step 1 - Convolution
classifier.add(layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(layers.Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# # Step 2 - Pooling

classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))

# Adding a second convolutional layer
classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))

classifier.add(layers.Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (3, 3)))


# Step 3 - Flattening
classifier.add(layers.Flatten())
#classifier.add(layers.Dropout(0.2))

# Step 4 - Full connection
classifier.add(layers.Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'relu'))
classifier.add(layers.Dense(number_classes, activation='sigmoid'))
#classifier.add(layers.Dense(units = 1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'D:\Python_code\SOURCE CODE/dataset',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'D:\Python_code\SOURCE CODE/dataset',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs =10,
                         validation_data = test_set,    
                         validation_steps = 50)

classifier.save("modelforcancer.h5")
print("Saved model to disk")
classifier.summary()

# from keras.layers import Conv2D, MaxPoling2D
# Implamentation of VGG16 

def VGG16():

    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu")),
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")),
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))),
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))),
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2))),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")),
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="vgg16")),
    model.add(Flatten(name="flatten")),
    model.add(Dense(256, activation="relu", name="fc1"))
    model.add(Dense(128, activation="relu", name="fc2"))
    model.add(Dense(196, activation="softmax", name="output"))
    return model


model=VGG16()
model.summary()
Vgg16 = model(inputs=model.input, outputs=model.get_layer('vgg16').output)







