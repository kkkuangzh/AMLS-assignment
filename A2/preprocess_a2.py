import pandas as pd
import numpy as np
import os

import face_recognition
from sklearn.model_selection import train_test_split


# prepare 72 features points of every images for KNN and SVM
def extract_facial_features_a2(img_path):
    face_locations_list = []
    face_landmarks_list = []

    for path, subpath, files in os.walk(img_path):
        for i in range(len(files)):
            name = img_path + str(i) + ".jpg"
            image = face_recognition.load_image_file(name)
            face_locations_list.append(face_recognition.face_locations(image))
            face_landmarks_list.append(face_recognition.face_landmarks(image))

    return face_landmarks_list, face_locations_list


# find 24 lips points from images that can be recognized as face
def recognized_images_a2(img_path, feature_list):
    face_landmarks_list, face_locations_list = extract_facial_features_a2(img_path)
    lips_features = []
    face_locations = []
    not_detected_labels = []
    for i in range(len(face_landmarks_list)):
        if face_landmarks_list[i] == []:
            not_detected_labels.append(i)
            continue
        else:
            temp = []
            temp += face_landmarks_list[i][0][feature_list[-1]]
            temp += face_landmarks_list[i][0][feature_list[-2]]
            lips_features.append(temp)
            face_locations.append(face_locations_list[i][0])

    lips_features = np.asarray(lips_features)
    face_locations = np.asarray(face_locations)

    return lips_features, face_locations, not_detected_labels


# normalise coordinates within its corresponding face
def normalise_features_a2(lips_features, face_locations):
    normalised_lips_features = []

    for i in range(len(lips_features)):
        origin_point = np.asarray([face_locations[i][-1], face_locations[i][0]])
        lips_features[i] = lips_features[i] - origin_point
        temp_max = lips_features[i].max(0)
        temp_min = lips_features[i].min(0)
        normalised_lips_features.append((lips_features[i] - temp_min) / (temp_max - temp_min))

    normalised_lips_features = np.asarray(normalised_lips_features)
    normalised_lips_features_flatten = normalised_lips_features.reshape(normalised_lips_features.shape[0], 48)

    print(len(normalised_lips_features_flatten), 'of images are detected and normalised.')
    return normalised_lips_features_flatten


# load original labels from directory
def load_taskA_labels_a2(label_path):
    labels = pd.read_csv(label_path)
    smiling = []

    for i in range(len(labels['\timg_name\tgender\tsmiling'])):
        smiling.append(labels['\timg_name\tgender\tsmiling'][i].split("\t")[3])

    smiling = np.asarray(smiling).astype("int")
    smiling[smiling == -1] = 0

    return smiling


# find labels of these recognized images
def find_detected_labels_a2(label_path, not_detected_labels):
    original_smiling_labels = load_taskA_labels_a2(label_path)
    smiling_detected = []

    for i in range(len(original_smiling_labels)):
        if i not in not_detected_labels:
            smiling_detected.append(original_smiling_labels[i])

    smiling_detected = np.asarray(smiling_detected)

    return smiling_detected


def preprocessing_a2(img_path, label_path):
    feature_list = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                    'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    face_features, face_locations, not_detected_labels = recognized_images_a2(img_path, feature_list)

    normalised_face_features_flatten = normalise_features_a2(face_features, face_locations)
    recongized_labels = find_detected_labels_a2(label_path, not_detected_labels)

    return normalised_face_features_flatten, recongized_labels


def get_a2_data(img_path, label_path):

    print('A2: Preparing lips features from directory for task A2...')
    normalised_lips_features_flatten, recongized_labels = preprocessing_a2(img_path, label_path)
    lips_feature_train, lips_feature_test, lips_feature_train_label, lips_feature_test_label = \
                                                        train_test_split(normalised_lips_features_flatten,
                                                        recongized_labels, test_size=0.2,
                                                        random_state=1)

    return lips_feature_train, lips_feature_test, lips_feature_train_label, lips_feature_test_label


# ************************** get additional test data *********************
def get_a2_test_data(test_img_path, test_label_path):

    normalised_face_features_flatten, recongized_labels = preprocessing_a2(test_img_path, test_label_path)

    return normalised_face_features_flatten, recongized_labels













