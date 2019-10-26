from keras.layers import Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, ELU, Dropout, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import tensorflow as tf
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing import image
import matplotlib.pyplot as plt

import csv
import cv2
import math
import numpy as np
import os


def load_data(path):

    lines = []
    with open(path + '/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    lines = lines[1:]
    return lines


def brightness(image):

    new_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_bright = np.random.uniform(0.1, 0.9)
    new_img[:, :, 2] = new_img[:, :, 2]*rand_bright
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    return new_img

def preprocess_img(img):
    new_img = img[50:140, :, :]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3, 3), 0)
    new_img = cv2.resize(new_img, (320, 160), interpolation = cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    return new_img

def random_distort(img, angle):
    """
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position """
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:, :, 0] + value) > 255
    if value <= 0:
        mask = (new_img[:, :, 0] + value) < 0
    new_img[:, :, 0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h, w = new_img.shape[0:2]
    mid = np.random.randint(0, w)
    factor = np.random.uniform(0.6, 0.8)
    if np.random.rand() > .5:
        new_img[:, 0:mid, 0] *= factor
    else:
        new_img[:, mid:w, 0] *= factor
    # randomly shift horizon
    h, w, _ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8, h/8)
    pts1 = np.float32([[0, horizon], [w, horizon], [0, h], [w, h]])
    pts2 = np.float32([[0, horizon+v_shift], [w, horizon+v_shift], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return new_img.astype(np.uint8), angle

def data_aug(images, measurements):

    aug_images, aug_measurements = [], []
    for image, measurement in zip(images, measurements):
        aug_images.append(image)
        aug_measurements.append(measurement)
        flip_img = cv2.flip(image, 1)
        flip_measurement = (measurement*-1)
        aug_images.append(flip_img)
        aug_measurements.append(flip_measurement)

        image_bright1 = brightness(image)
        image_bright2 = brightness(flip_img)

        aug_images.append(image_bright1)
        aug_measurements.append(measurement)
        aug_images.append(image_bright2)
        aug_measurements.append(flip_measurement)

    return aug_images, aug_measurements

def balance_data(samples, visulization_flag ,N=60, K=1,  bins=100):
    """ Crop the top part of the steering angle histogram, by removing some images belong to those steering angels
    :param images: images arrays
    :param angles: angles arrays which
    :param n:  The values of the histogram bins
    :param bins: The edges of the bins. Length nbins + 1
    :param K: maximum number of max bins to be cropped
    :param N: the max number of the images which will be used for the bin
    :return: images, angle
    """

    angles = []
    for line in samples:
        angles.append(float(line[3]))

    n, bins, patches = plt.hist(angles, bins=bins, color= 'orange', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)

    index = n.argsort()[-K:][::-1]    # find the largest K bins
    del_index = []                    # collect the index which will be removed from the data
    for i in range(K):
        if n[index[i]] > N:
            ind = np.where((bins[index[i]]<=angles) & (angles<bins[index[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            del_index.extend(ind[:len(ind)-N])

    # angles = np.delete(angles,del_index)
    balanced_samples = [v for i, v in enumerate(samples) if i not in del_index]
    balanced_angles = np.delete(angles,del_index)

    plt.subplot(1,2,2)
    plt.hist(balanced_angles, bins=bins, color= 'orange', linewidth=0.1)
    plt.title('modified histogram', fontsize=20)
    plt.xlabel('steering angle', fontsize=20)
    plt.ylabel('counts', fontsize=20)

    if visulization_flag:
        plt.figure
        plt.subplot(1,2,1)
        n, bins, patches = plt.hist(angles, bins=bins, color='orange', linewidth=0.1)
        plt.title('origin histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

        plt.figure
        aa = np.append(balanced_angles, -balanced_angles)
        bb = np.append(aa, aa)
        plt.hist(bb, bins=bins, color='orange', linewidth=0.1)
        plt.title('final histogram', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

    return balanced_samples

def conv_net():

    model = Sequential()
    model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160, 320, 3)))
    # Crop out the top pixels and bottom pixels, no cropping on left and right pixels
    model.add(Cropping2D(cropping=((68, 20), (0, 0))))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    filename = batch_sample[i].split('/')[-1]
                    source_path = (path + '/data/IMG/' + filename)
                    image = cv2.imread(source_path)
                    image, measurement = random_distort(preprocess_img(image), batch_sample[i])
                    images.append(image)
                    measurement = float(batch_sample[3])
                    measurements.append(measurement)
                    if abs(measurement) >= 0:
                        corr = 0.3
                    if i == 1:
                        measurement = measurement + corr
                    if i == 2:
                        measurement = measurement - corr
                    images.append(image)
                    measurements.append(measurement)

            aug_images, aug_measurements = data_aug(images, measurements)

            X_train = np.array(aug_images)
            y_train = np.array(aug_measurements)
            yield shuffle(X_train, y_train)


path = '/Users/p.dot/Desktop'

samples = load_data(path)
train_samples, valid_samples = train_test_split(
    samples, shuffle=True, test_size=0.2)

# balance the data with smooth the histogram of steering angles
samples = balance_data(samples, visulization_flag=True)

train_gen = generator(train_samples, batch_size=32)
valid_gen = generator(valid_samples, batch_size=32)

batch_size = 32
model = conv_net()
model.summary()

model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_gen, samples_per_epoch=len(train_samples)//2,
                    validation_data=valid_gen, nb_val_samples=len(valid_samples), nb_epoch=15)

model.save('model.h5')
print("Model Saved")

