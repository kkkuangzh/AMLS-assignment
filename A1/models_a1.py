from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from preprocess_a1 import get_test_data_a1


# knn on 72 feature points
def knn_model_a1(x_train, x_test, y_train, y_test, k=4, mode='predict'):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    if mode == 'grid_search':
        return score

    print('A1: Accuracy of KNN on split test set is: ', score)
    return score, neigh


# find the best parameter k for knn
def grid_search_for_knn_a1(x_train, x_test, y_train, y_test, search_num):

    knn_scores = []

    for i in range(1, search_num):
        score = knn_model_a1(x_train, x_test, y_train, y_test, i, mode='grid_search')
        knn_scores.append(score)

    print("A1: The best k for knn is the " + str(np.argmax(knn_scores)) +
          " and best acc is " + str(np.max(knn_scores)))

    return knn_scores


# svm on 72 feature points
def svm_model_a1(x_train, x_test, y_train, y_test):
    # kernel='linear' poly rbf
    clf = SVC(kernel='poly', C=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    print("A1: Accuracy of SVM on split test set is: ", score)

    return score, clf


# find best parameters for svm
def grid_search_for_svm_a1(x_train, x_test, y_train, cv_folds):
    tuned_parameters = [{'kernel': ['rbf'],
                         'C': [0.1, 1, 10],
                         'gamma': [0.1, 1, 10]},
                        {'kernel': ['linear'],
                         'C': [0.1, 1, 10]},
                        {'kernel': ['poly'],
                         'C': [0.1, 1, 10]}]

    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=kfold, scoring='precision', verbose=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    for para, mean_score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
        print(para, mean_score)

    return y_pred


# ********************* CNN model *********************
def define_generator_a1(x_train, y_train):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=16,
    )
    return train_generator


# Train on InceptionV3 Model
def define_cnn_model_a1():
    base_model = InceptionV3(weights='./A1/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                             include_top=False,
                             input_shape=(218, 178, 3))

    base_model.trainable = False

    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(GlobalAveragePooling2D())
    add_model.add(Dense(1, activation='sigmoid'))

    model = add_model
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_cnn_model_a1(x_train, y_train, x_valid, y_valid):
    print("A1: Training CNN model based on Inception-v3 model...")
    train_generator = define_generator_a1(x_train, y_train)
    model = define_cnn_model_a1()
    # checkpointer = ModelCheckpoint(filepath='A1_transfer_test_augment.h5',
    #                                verbose=1, save_best_only=True)

    hist = model.fit_generator(train_generator
                               , validation_data=(x_valid, y_valid)
                               , steps_per_epoch=200
                               , epochs=10
                               # , callbacks=[checkpointer]
                               , verbose=1
                               )
    print("A1: Model has been trained successfully without saving!")
    return hist


def evaluate_cnn_model_a1(model_path, x_test, y_test):

    model = load_model(model_path)
    acc = model.evaluate(x_test, y_test, verbose=1)

    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] >= 0.5] = 1
        y_pred[i][y_pred[i] < 0.5] = 0

    print('A1: Accuracy of CNN on split test set is: ', acc[1])
    target_names = ['male', 'female']
    print('\n', classification_report(y_test, y_pred, digits=3, target_names=target_names))

    return acc[1]


def acc_on_additional_test_set_a1(test_img_path, test_label_path, model_path, knn_clf, svm_clf):
    normalised_face_features_flatten, recongized_labels, imgs_standardized, gender_label \
        = get_test_data_a1(test_img_path, test_label_path)

    y_pred_knn = knn_clf.predict(normalised_face_features_flatten)
    acc_of_knn = accuracy_score(recongized_labels, y_pred_knn)

    y_pred_svm = svm_clf.predict(normalised_face_features_flatten)
    acc_of_svm = accuracy_score(recongized_labels, y_pred_svm)

    cnn_model = load_model(model_path)
    acc_of_cnn = cnn_model.evaluate(imgs_standardized, gender_label, verbose=1)

    print('\nA1: Accuracy of KNN, SVM, CNN on additional test set are', acc_of_knn, acc_of_svm, acc_of_cnn[1], ' respectively.')
    return acc_of_knn, acc_of_svm, acc_of_cnn[1]




