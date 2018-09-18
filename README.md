# OVERVIEW

This Repository showcases a project on Image Classification.
Images are classified in 5 different categories by Training the network. The model is a perfect example of supervised Classification problem.

It uses Transfer learning wherein a pretrained model is used for a set of categories like ImageNet, and then retrained from the existing weights for new classes. 

The changes made here include retraining the network through InceptionV3 model by Google. It can also be trained through MobileNet.

This repository uses codetensorflow-for-poets-2 from the "TensorFlow for poets 2" series of codelabs.

SETUP:

* Install TensorFlow:

  This model will use TensorFlow library. You can learn more about it on https://en.wikipedia.org/wiki/TensorFlow. Install it by the following command.
  
   terminal: pip install tensorflow
   
   As the codelab was tested on TensorFlow version 1.7, upgrade it by:
   
   terminal:  pip install --upgrade "tensorflow==1.7.*"
   
* Clone the git repository:
  
  This repository contains the code used in this codelab. By cloning the repository, you will get to the platform where we will be working. Copy the code in your terminal.

   git clone 
   
   cd tensorflow-for-poets-2

* Download the traning images:
 
  Before training the model, you would require a set of images to teach the model about the new classes of flowers it should recognize.

  http://download.tensorflow.org/example_images/flower_photos.tgz
  
  To download these images, copy the above link to new tab of your browser or type the above commands in your terminal:

  curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files

* Retrain the network:
  
  The network is retrained through Inception V3 model. I am not using MobileNet, though its small and effiecient, Because its accuracy is lower than Inception V3. The scripts downloads the pre-trained network and then after adding a last layer of classification onto it, we train it on our images and provide our labels. To start retraining the network, copy the following code on your terminal.
  
  python -m scripts.retrain \
    --bottleneck_dir=tf_files/bottlenecks \
    --model_dir=tf_files/models \
    --summaries_dir=tf_files/training_summaries \
    --output_graph=tf_files/retrained_graph.pb \
    --how_many_training_steps=4000 \
    --output_labels=tf_files/retrained_labels.txt \
    --architecture=inception_v3 \
    --image_dir=tf_files/flower_photos \
    
  
  ImageNet does not include any of these flower species we're training on here. However, the kinds of information that make it possible for ImageNet to differentiate among 1,000 classes are also useful for distinguishing other objects. By using this pre-trained network, we are using that information as input to the final classification layer that distinguishes our flower classes.
  


There are multiple versions of this codelab depending on which version of the tensorflow libraries you plan on using:

* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).


This repo contains simplified and trimmed down version of tensorflow's example image classification apps.

* The TensorFlow Lite version, in `android/tflite`, comes from [tensorflow/contrib/lite/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite).
* The Tensorflow Mobile version, in `android/tfmobile`, comes from [tensorflow/examples/android/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).

The `scripts` directory contains helpers for the codelab. Some of these come from the main TensorFlow repository, and are included here so you can use them without also downloading the main TensorFlow repo (they are not part of the TensorFlow `pip` installation).

