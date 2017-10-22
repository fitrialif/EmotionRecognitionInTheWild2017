from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
import numpy as np
import dlib
from skimage import io
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import patches




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

def normalize(img):
    img = img / 255.0
    return img

if __name__ == "__main__":
    
    detector = dlib.get_frontal_face_detector()
    model = Hvgg()
    model.load_weights('best_model2.hdf5')
    path = '/path/to/image/'
    imagename = '1.jpg'
    img = io.imread(path+imagename)
    Emotions = ['Negative', 'Neutral', 'Positive']
    
    fig,ax = plt.subplots(1)
    ax.imshow(img) 

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        face = img[d.top():d.bottom(),d.left():d.right(),:]
        face = resize(face,(224,224))  
        face = np.expand_dims(face,0)
        em = model.predict(face)[0]
        em = np.argmax(em)
        
        Width = abs(d.top()-d.bottom())
        He = abs(d.left()-d.right())
        rect = patches.Rectangle((d.left(),d.top()),Width,He,linewidth=1,edgecolor='r',facecolor='none')
        ax.text(d.left(),d.top(),Emotions[em],fontsize=15, color='red')
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(imagename)   
    plt.show()
        
    