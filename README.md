# OVERVIEW

This Repository showcases a project on Image Classification.
Images are classified in 5 different categories by Training the network. The model is a perfect example of supervised Classification problem.

It uses Transfer learning wherein a pretrained model is used for a set of categories like ImageNet, and then retrained from the existing weights for new classes. 

The changes made here include retraining the network through InceptionV3 model by Google. It can also be trained through MobileNet.

This repository uses codetensorflow-for-poets-2 from the "TensorFlow for poets 2" series of codelabs.

SETUP:

* Install TensorFlow:

  This model will use TensorFlow library. You can learn more about it on https://en.wikipedia.org/wiki/TensorFlow. Install it by the following command.
  
   terminal: `pip install tensorflow`
   
   As the codelab was tested on TensorFlow version 1.7, upgrade it by:
   
   terminal: `pip install --upgrade "tensorflow==1.7.*"`
   
* Clone the git repository:
  
  This repository contains the code used in this codelab. By cloning the repository, you will get to the platform where we will be working. Copy the code in your terminal.

   ``git clone https://github.com/soumya2103/tensorflow-for-poets-2
   
   cd tensorflow-for-poets-2``

* Download the traning images:
 
  Before training the model, you would require a set of images to teach the model about the new classes of flowers it should recognize.

  http://download.tensorflow.org/example_images/flower_photos.tgz
  
  To download these images, copy the above link to new tab of your browser or type the above commands in your terminal:

  ``curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files``

* Retrain the network:
  
  The network is retrained through Inception V3 model. I am not using MobileNet, though its small and effiecient, Because its accuracy is lower than Inception V3. The scripts downloads the pre-trained network and then after adding a last layer of classification onto it, we train it on our images and provide our labels. To start retraining the network, copy the following code on your terminal.
  
  ```python -m scripts.retrain \
    --bottleneck_dir=tf_files/bottlenecks \
    --model_dir=tf_files/models \
    --summaries_dir=tf_files/training_summaries \
    --output_graph=tf_files/retrained_graph.pb \
    --how_many_training_steps=4000 \
    --output_labels=tf_files/retrained_labels.txt \
    --architecture=inception_v3 \
    --image_dir=tf_files/flower_photos \
   ``` 
    
  ImageNet does not include any of these flower species we're training on here. However, the kinds of information that make it possible for ImageNet to differentiate among 1,000 classes are also useful for distinguishing other objects. By using this pre-trained network, we are using that information as input to the final classification layer that distinguishes our flower classes.

  The script runs 4,000 training steps. Each step chooses 10 images at random from the training set, finds their bottlenecks from the cache, and feeds them into the final layer to get predictions. Those predictions are then compared against the actual labels, and the results of this comparison is used to update the final layer's weights through a backpropagation process. 
  
 * Using Retrained Model:
   The retrained scripts forms two files:
   
    * tf_files/retrained_graph.pb, which contains a version of the selected network with a final layer retrained on your categories.
    * tf_files/retrained_labels.txt, which is a text file containing labels.
   
   The repository has label_image.py which can be used to test the network.For classifying images, copy the following code into your terminal with your input image attached to it. 
   
   ```python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=[input image]
   ```
   you will obtain results indicating confidence levels of different labels and in this way highest confidence label would be your desired result.
