import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model


# extract face ROI of each image and normalise
def extract_eye_color_roi(img_path):
    eye_roi = []

    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            img = np.asarray(Image.open(img_path + str(i) + ".png").convert("RGB"))
            eye_roi.append(img[235:290, 170:240, :])

    eye_roi = np.asarray(eye_roi)
    normalised_eye_roi = eye_roi / 255

    return normalised_eye_roi


# load original labels from directory
def load_eye_color_labels(label_path):

    labels = pd.read_csv(label_path)
    eye_color = []
    for i in range(len(labels['\teye_color\tface_shape\tfile_name'])):
        eye_color.append(labels['\teye_color\tface_shape\tfile_name'][i].split("\t")[1])

    eye_color = np.asarray(eye_color).astype("int")

    return eye_color


def eye_color_label_onehot(label_path):

    eye_color = load_eye_color_labels(label_path)
    onehot_encoder = OneHotEncoder(sparse=False)
    eye_color_onehot = onehot_encoder.fit_transform(eye_color.reshape(-1, 1))

    return eye_color_onehot


def get_b2_data(img_path, label_path):
    print('B2: Extracting ROI of left eye from images and normalizing for task B2...')
    eye_color_roi = extract_eye_color_roi(img_path)
    eye_color_onehot = eye_color_label_onehot(label_path)
    print('B2: Labels changed to 5d one-hot encoding.')
    x_train, x_test, y_train, y_test = train_test_split(eye_color_roi, eye_color_onehot,
                                                        test_size=0.2, random_state=3)

    return x_train, x_test, y_train, y_test


def get_test_data_b2(test_img_path, test_label_path):

    eye_color_roi = extract_eye_color_roi(test_img_path)
    eye_color_onehot = eye_color_label_onehot(test_label_path)

    return eye_color_roi, eye_color_onehot


# 只有1076一张3没分出来
def add_class_6_to_label(model_path, img_path, label_path):
    print("B2: Adding sunglasses as a new class to labels...")
    eye_color_roi = extract_eye_color_roi(img_path)
    eye_color = load_eye_color_labels(label_path)
    model_b2 = load_model(model_path)
    y_pred = model_b2.predict(eye_color_roi)

    y_pred_list = []

    # onehot to number
    for i in range(len(y_pred)):
        y_pred_list.append(np.argmax(y_pred[i]))

    y_pred_list = np.asarray(y_pred_list)

    # find index of label 3
    class_3_pred = []
    class_3_truth = []

    for i in range(len(y_pred_list)):
        if y_pred_list[i] == 3:
            class_3_pred.append(i)
        if eye_color[i] == 3:
            class_3_truth.append(i)

    class_3_pred = np.asarray(class_3_pred)
    class_3_truth = np.asarray(class_3_truth)

    wrong_3 = [i for i in class_3_pred if i not in class_3_truth]

    eye_color[wrong_3] = 5

    onehot_encoder = OneHotEncoder(sparse=False)
    eye_color_onthot = onehot_encoder.fit_transform(eye_color.reshape(-1, 1))

    # return 6-dimension vector
    print('B2: Labels changed to from 5d to 6d one-hot encoding.')
    return eye_color_roi, eye_color_onthot




