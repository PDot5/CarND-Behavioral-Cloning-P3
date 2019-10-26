# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The repository above contains starting files for the Behavioral Cloning Project.

In this project, the goal is to apply what has been taught in regards to deep learning networks and convolutional neural networks in order to clone driving behavior. Using Keras, I will train, validate, and test a model. The model will output a steering angle to an autonomous vehicle.

Using the simulator that has already been provided, I'll be able to use it in order to collect data for image processing as well as steering angles data in order to train the neural network. This data will be used to help build a neural network that teaches the vehicle to drive autonomously around the track.

# Objectives:

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Details About Files In This Directory

### `model.py`
This is where the training and processing data can be found, as well as the model architecture. Processing functions include:
* load_data function: In this function, we call in the data collected from the steering column and this will be used with the convolutional network to train and validate the data.
* brightness function: Randomly change the brightness between predetermined thresholds.
* data_aug function: During data augmentation, we flip the images on the horizontal axis which gives double the amount of data. The brightness function is called on which is also applied to the flipped images. This data is then appended to an array for augmented images and augmented measurements.
* generator function: Batch samples of images are read in and appended to a list. The measurements are converted to float for the steering column and through the process of machine learning, the correction factor of (0.2) is set for the vehicle to autonomously make corrections to steering for either left (+ corr) of right (- corr) of center.
* preprocess_img function: This created a new image of a selected region in the y coordinates and the x coordinates then I applied a gaussianblur. Next I resized the image and convered the color.
* random_dist function: Adding random distortion to dataset images as well as random brightness adjustments plus a random vertical shift of the horizon position.
* balance_data function: This will crop the top part of the steering angle histogram. It will do this by removing some images belonging to those steering angels. 
* conv_net function: This function holds the model architecture.

## Architecture and Training

| Layers                    | Description       |
| --------------------------|:-------------------:|
|Model Sequential           | Linear Stack of Layers |
| Lambda                    | Helps implement layers or functionality that is not prebuilt and which do not require trainable weights. Input_shape: (160, 320, 3) |
| Cropping (68, 20), (0,0)  | Crop the image to cut out upper pixels (tree and sky region) as well as bottom pixels (hood of vehicle), leaving the sides set to zero |
| Convolution 1:            | 24 Filters, kernal 5x5, activation: elu, strides 2x2 |
| Convolution 2:            | 36 Filters, kernal 5x5, activation: elu, strides 2x2  |
| Convolution 2:            | 48 Filters, kernal 5x5, activation: elu, strides 2x2  |
| Add Pooling               | Max Pooling dimensions (2x2) |
| Add Flatten               | Flatten Convolution 2 Input 5X5X16 Output 400 |
| Add Dense                 | Fully Connected: (512) |
| Add Dropout               | For Overfitting, Dropout: 0.2 |
| Add Dense                 | Fully Connected: (128) |
| Add Dropout               | For Overfitting, Dropout: 0.2 |
| Add Dense                 | Fully Connected: (10) |
| Add Dense                 | Fully Connected: (1) |

* train_test_split: Using this we can easily split the dataset into the training and the testing datasets in various proportions. The samples are shuffled and the test size was set to 0.2.
* fit_generator function: This function is called and passed parameters such as train_gen which is generated from the training samples and a determined batch_size, ie: 32. The samples_per_epoch is collected from the length of the train_samples, the validation_generator is set to valid_gen which is also generated from the valid samples, nb_val_samples is determined by the length of the valid_samples, and nb_epoch is set to a predetermined number, ie: 3.

### Training
* During the training process, I found more methods were needed in order to better detect the boundaries of the lane lines. Creating a new image with specified coordinates as well as balancing the data set and applying a gaussian blur seemed to help resolve the training issues I ran into previously. I noticed that the vehicle, although driving pretty well, it would tent to take the dirt path after the bridge rather then detect the boundary and make the correct adjustments to stay on the road. Using the newly added methods, I was able to better train the model in order to meet the requirements of the project. It is not pretty, but it remains on the road. I ran low on GPU time which is why I have added my local computer output-video in order to demonstrate the ability of my convolutional network:

[![Alternate Text]({https://github.com/PDot5/CarND-Behavioral-Cloning-P3/blob/master/center_2019_10_19_00_30_25_004.jpg})]({https://youtu.be/B3RCtNCnHX8} "Local Computer Output Video")

###Histogram Output
![alt text](https://github.com/PDot5/CarND-Behavioral-Cloning-P3/blob/master/Histogram.png "Original and Modified Histogram")

![alt text](https://github.com/PDot5/CarND-Behavioral-Cloning-P3/blob/master/Final_Histogram.png "Final Histogram")

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```
## Potential Shortcomings
The ability of GPU while using overseas in Afghanstan may have hindered the epoch training time. At times the system would take several hours and the GPU workspace would go to sleep. Countless hours were lost trying to train the data.

Additionally, the time for each epoch had to be serverely reduced in order to complete the project. This was done by dividing the number of samples_per_epoch by dividing the training samples by a factor of 3. The number of epochs had to also be cut down in order to achieve a training set to use on the autonomous vehicle in a reasonable time period. Increasing the EPOCHS will give a greater driving ability as I have tested on my local computer with a samples_per_epoch divided by a factor of 2 and EPOCH set to 15. I have uploaded the local_comp_output_video.mp4 in order to demonstrate the better performance.

Another shortcoming was definitely in gathering the training data. It took several hours for me, based on the slow connection speed overseas, in order to collect viable data. The lag of the simulation caused a lot of grief in order to get the vehicle somewhat safely around the track.

## Suggest possible improvements to your pipeline
* Adding additional processing techniques such as gradient thresholds, magnitude thresholds, convert to HLS, and canny edge detector could help the autonomous vehicle training set by determing the edge of the road and/or bridge.
* Using a second or even third set of data. Unfortunatley this is not an option in my current location due to the lag time in the simulator.
* Determing best Dropout Rate
* Increasing training set.

### References:
# https://www.programcreek.com/python/example/89703/keras.layers.Cropping2D
# https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
# https://keras.io/layers/advanced-activations/
# https://github.com/cdemutiis/Behavioral-Cloning/blob/master/model.py
# https://github.com/shuklam20/CarND-Behavioral-Cloning-P3/blob/master/model.py
# https://github.com/jayshah19949596/Behavioral-Cloning-For-Self-Driving-Car/blob/master/Final_BC.ipynb
# https://github.com/JunshengFu/driving-behavioral-cloning/blob/master/model.py


