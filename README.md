# EmotionRecognitionInTheWild2017
A ConvNet Model for recogntion of emotion.
Dataset is available at https://1drv.ms/f/s!AvCvxAqN76_2gux9t878Vo87BrkJkw

# Prerequisites
- Keras
- Tensorflow
- Numpy
- Matplotlib
- cv2
- dlib

# How to use
In order to prepare the data you need to first run the extractFaces script for each folder of dataset to extract only faces.Then you need to run The Train Script with the right reffrence for data location. After the training is done. you run the prediction script by fixing the location of the image and the saved model.
```
path = '/path/to/image/'
imagename = '1.jpg'
```
# Prediction Samples
The trained model can be found here: https://drive.google.com/drive/folders/0B5rYsA28q_6cY0szOXVCYmJBYkU?usp=sharing
here is some examples of predictions:
<p align="center">
  <img src="images/2.jpg" />
  <img src="images/3.jpg" />
  <img src="images/4.jpg" />
  <img src="images/5.png" />
  <img src="images/6.png" />
</p>
