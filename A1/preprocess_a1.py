import pandas as pd
import numpy as np
import os
from PIL import Image

import face_recognition
from sklearn.model_selection import train_test_split


# prepare 72 features points of every images for KNN and SVM
def extract_facial_features_a1(img_path):
    face_locations_list = []
    face_landmarks_list = []

    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            name = img_path + str(i) + ".jpg"
            image = face_recognition.load_image_file(name)
            face_locations_list.append(face_recognition.face_locations(image))
            face_landmarks_list.append(face_recognition.face_landmarks(image))

    return face_landmarks_list, face_locations_list


# find those images that can be recognized as face
def recognized_images_a1(img_path, feature_list):
    face_landmarks_list, face_locations_list = extract_facial_features_a1(img_path)
    face_features = []
    face_locations = []
    not_detected_labels = []

    for i in range(len(face_landmarks_list)):
        if face_landmarks_list[i] == []:
            not_detected_labels.append(i)
            continue
        else:
            temp = []
            for j in range(len(feature_list)):
                temp += face_landmarks_list[i][0][feature_list[j]]
            face_features.append(temp)
            face_locations.append(face_locations_list[i][0])

    face_features = np.asarray(face_features)
    face_locations = np.asarray(face_locations)

    return face_features, face_locations, not_detected_labels


# normalise coordinates within its corresponding face
def normalise_features_a1(face_features, face_locations):
    normalised_face_features = []
    for i in range(len(face_features)):
        origin_point = np.asarray([face_locations[i][-1], face_locations[i][0]])
        face_features[i] = face_features[i] - origin_point
        temp_max = face_features[i].max(0)
        temp_min = face_features[i].min(0)
        normalised_face_features.append((face_features[i] - temp_min) / (temp_max - temp_min))

    normalised_face_features = np.asarray(normalised_face_features)
    normalised_face_features_flatten = normalised_face_features.reshape(normalised_face_features.shape[0], 144)
    print(len(normalised_face_features_flatten), 'of images are detected and normalised.')

    return normalised_face_features_flatten


# load original labels from directory
def load_taskA_labels(label_path):
    labels = pd.read_csv(label_path)
    gender = []

    for i in range(len(labels['\timg_name\tgender\tsmiling'])):
        gender.append(labels['\timg_name\tgender\tsmiling'][i].split("\t")[2])

    gender = np.asarray(gender).astype("int")
    gender[gender == -1] = 0

    return gender


# find labels of these recognized images
def find_detected_labels_a1(label_path, not_detected_labels):
    original_gender_labels = load_taskA_labels(label_path)
    gender_detected = []

    for i in range(len(original_gender_labels)):
        if i not in not_detected_labels:
            gender_detected.append(original_gender_labels[i])

    gender_detected = np.asarray(gender_detected)

    return gender_detected


def preprocessing_a1(img_path, label_path):
    feature_list = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                    'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    face_features, face_locations, not_detected_labels = recognized_images_a1(img_path, feature_list)

    normalised_face_features_flatten = normalise_features_a1(face_features, face_locations)
    recongized_labels = find_detected_labels_a1(label_path, not_detected_labels)

    return normalised_face_features_flatten, recongized_labels


def get_a1_data(img_path, label_path):
    print('Preparing facial features for KNN and SVM of A1...')
    normalised_face_features_flatten, recongized_labels = preprocessing_a1(img_path, label_path)
    face_feature_train, face_feature_test, face_feature_train_label, face_feature_test_label = \
                                                        train_test_split(normalised_face_features_flatten,
                                                        recongized_labels, test_size=0.2,
                                                        random_state=23)

    return face_feature_train, face_feature_test, face_feature_train_label, face_feature_test_label


# ************************** preprocessing for CNN *********************
# load images from directory for CNN
def load_taskA_images_a1(img_path):
    imgs = []
    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            img = Image.open(img_path + str(i) + ".jpg")
            imgs.append(np.asarray(img))
    imgs = np.asarray(imgs)
    imgs_standardized = imgs / 255
    return imgs_standardized


def get_a1_data_cnn(img_path, label_path):
    print('A1: Loading images from directory and normalising..')
    imgs_standardized = load_taskA_images_a1(img_path)
    gender = load_taskA_labels(label_path)

    x_train, x_test, y_train, y_test = train_test_split(imgs_standardized, gender, test_size=0.2, random_state=23)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=13)

    return x_train, x_test, x_valid, y_train, y_test, y_valid


# ************************** get additional test data *********************
def get_test_data_a1(test_img_path, test_label_path):
    # print('Load images and features from additional test set...')
    normalised_face_features_flatten, recongized_labels = preprocessing_a1(test_img_path, test_label_path)
    imgs_standardized = load_taskA_images_a1(test_img_path)
    gender_label = load_taskA_labels(test_label_path)

    return normalised_face_features_flatten, recongized_labels, imgs_standardized, gender_label



