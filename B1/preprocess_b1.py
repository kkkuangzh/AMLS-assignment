import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# extract face ROI of each image and normalise
def extract_face_shape_roi(img_path):
    face_shape_roi = []

    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            img = np.asarray(Image.open(img_path + str(i) + ".png").convert("RGB"))
            face_shape_roi.append(img[300:400, 150:350, :])

    face_shape_roi = np.asarray(face_shape_roi)
    normalised_face_shape_roi = face_shape_roi / 255

    return normalised_face_shape_roi


# load original labels from directory
def load_face_shape_labels(label_path):
    labels = pd.read_csv(label_path)

    face_shape = []

    for i in range(len(labels['\teye_color\tface_shape\tfile_name'])):
        face_shape.append(labels['\teye_color\tface_shape\tfile_name'][i].split("\t")[2])

    face_shape = np.asarray(face_shape).astype("int")
    return face_shape


def face_shape_label_onehot(label_path):

    face_shape = load_face_shape_labels(label_path)
    onehot_encoder = OneHotEncoder(sparse=False)
    faces_shape_onthot = onehot_encoder.fit_transform(face_shape.reshape(-1, 1))
    return faces_shape_onthot


def get_b1_data(img_path, label_path):
    print('B1: Extracting ROI of lower part of the face from images and normalising for task B1... ')
    face_shape_roi = extract_face_shape_roi(img_path)
    face_shape_onehot = face_shape_label_onehot(label_path)
    print('B1: Labels changed to one-hot encoding.')
    x_train, x_test, y_train, y_test = train_test_split(face_shape_roi, face_shape_onehot,
                                                        test_size=0.2, random_state=3)

    return x_train, x_test, y_train, y_test


def get_test_data_b1(test_img_path, test_label_path):
    face_shape_roi = extract_face_shape_roi(test_img_path)
    face_shape_onehot = face_shape_label_onehot(test_label_path)

    return face_shape_roi, face_shape_onehot




