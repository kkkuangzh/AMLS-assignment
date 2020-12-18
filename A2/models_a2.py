from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import numpy as np
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, load_model
from preprocess_a2 import get_a2_test_data


# knn on 24 feature points
def knn_model_a2(x_train, x_test, y_train, y_test, k=11, mode='predict'):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    if mode == 'grid_search':
        return score

    print('A2: Accuracy of KNN on split test set is: ', score)
    return score, neigh


# find the best parameter k for knn
def grid_search_for_knn_a2(x_train, x_test, y_train, y_test, search_num):

    knn_scores = []

    for i in range(1, search_num):
        score = knn_model_a2(x_train, x_test, y_train, y_test, i, mode='grid_search')
        knn_scores.append(score)

    print("A2: The best k for knn is the " + str(np.argmax(knn_scores)) +
          " and best acc is " + str(np.max(knn_scores)))

    return knn_scores


# svm on 24 feature points
def svm_model_a2(x_train, x_test, y_train, y_test):
    # kernel='linear' poly rbf
    clf = SVC(kernel='rbf', C=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    print("A2: Accuracy of SVM on split test set is: ", score)

    return score, clf


# find best parameters for svm
def grid_search_for_svm_a2(x_train, x_test, y_train, cv_folds):
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


def define_dnn_model_a2():
    inputs = Input(shape=48)

    fcn1 = Dense(2048, activation='relu')(inputs)
    fcn2 = Dense(2048, activation='relu')(fcn1)
    outputs = Dense(1, activation='sigmoid')(fcn2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model


def train_dnn_model_a2(x_train, y_train):
    print("A2: Training DNN model...")
    model = define_dnn_model_a2()
    # checkpointer = ModelCheckpoint(filepath='A2_smiling_72_face_orig_normalizaed.h5',
    #                                verbose=1, save_best_only=True)

    smiling_history = model.fit(x_train, y_train, epochs=300,
                                batch_size=16,
                                # callbacks=[checkpointer],
                                verbose=1,
                                shuffle=True,
                                validation_split=0.2)

    print("A2: Model has been trained successfully")
    return smiling_history


def evaluate_dnn_model_a2(model_path, x_test, y_test):

    model = load_model(model_path)
    acc = model.evaluate(x_test, y_test, verbose=1)

    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] >= 0.5] = 1
        y_pred[i][y_pred[i] < 0.5] = 0

    print("A2: Accuracy of DNN on split test set is: ", acc[1])
    target_names = ['Smiling', 'Not smiling']
    print('\n', classification_report(y_test, y_pred, digits=3, target_names=target_names))

    return acc[1]


def acc_on_additional_test_set_a2(test_img_path, test_label_path, model_path, knn_clf, svm_clf):
    normalised_face_features_flatten, recongized_labels = get_a2_test_data(test_img_path, test_label_path)

    y_pred_knn = knn_clf.predict(normalised_face_features_flatten)
    acc_of_knn = accuracy_score(recongized_labels, y_pred_knn)

    y_pred_svm = svm_clf.predict(normalised_face_features_flatten)
    acc_of_svm = accuracy_score(recongized_labels, y_pred_svm)

    dnn_model = load_model(model_path)
    acc_of_dnn = dnn_model.evaluate(normalised_face_features_flatten, recongized_labels, verbose=1)

    print('\nA2: Accuracy of KNN, SVM, DNN on additional test set are', acc_of_knn, acc_of_svm, acc_of_dnn[1],
          'respectively.')
    return acc_of_knn, acc_of_svm, acc_of_dnn[1]




















