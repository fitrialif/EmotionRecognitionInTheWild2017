from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam,Adamax
import numpy as np
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
    
def Hvgg():
    img_input = Input(shape=(224,224,3))
    # Block 1
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal', name='block1_conv1')(img_input)
    x = MaxPooling2D((4, 4), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal', name='block1_conv2')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2), name='block2_pool')(x)   
    x = Conv2D(16, (3, 3), activation='relu', padding='same' ,kernel_initializer='he_normal', name='block3_conv1')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2), name='block3_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu',kernel_initializer='he_normal', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax',kernel_initializer='he_normal', name='predictions')(x)
    
    model = Model(img_input, x, name='hvgg')    
    print model.summary()
    return model


if __name__ == "__main__":
    
    batch_size = 32
    epochs = 50
    
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow_from_directory(
        'DATA2/Train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
    
    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
        'DATA2/Val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')  
    
    test_datagen = ImageDataGenerator(rescale=1./255)    
    
    
    model = Hvgg()#VGG_16()
    sgd = Adam(lr=0.0001, beta_1=0.5) #Adamax(lr=0.000002)#SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)#
    #Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.) ###
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    mcp = ModelCheckpoint(filepath='./best_model2.hdf5', verbose=1,monitor='val_loss',save_best_only=True)
    mcsv = CSVLogger('modelTracker2.csv', separator=',', append=False)
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=200,
        callbacks = [mcp,mcsv]
    )    