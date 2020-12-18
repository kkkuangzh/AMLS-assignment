from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from preprocess_b2 import get_test_data_b2
from preprocess_b2 import add_class_6_to_label


def B2_network():
    inputs = Input(shape=(55, 70, 3))
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    flatten1 = Flatten()(pool2)
    dense1 = Dense(16, activation='relu')(flatten1)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(5, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model


# train face_shape cnn
def train_cnn_model_b2(x_train, y_train):
    print("B2: Training B2 5 classes cnn model...")
    model = B2_network()
    # checkpointer = ModelCheckpoint(filepath='B2_eye_color_ROI_cnn.h5', verbose=1, save_best_only=True)
    eye_color_history = model.fit(x_train, y_train, epochs=10,
                                  batch_size=16,
                                  # callbacks=[checkpointer],
                                  verbose=1,
                                  shuffle=True,
                                  validation_split=0.2)

    print("B2: 5d CNN model has been trained successfully")
    return eye_color_history


def evaluate_cnn_model_b2(model_path, x_test, y_test):

    model_b2 = load_model(model_path)
    acc = model_b2.evaluate(x_test, y_test, verbose=1)

    y_pred = model_b2.predict(x_test)

    print('B2: Accuracy of 5d-CNN on split test data is:', acc[1])
    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] == max(y_pred[i])] = 1
        y_pred[i][y_pred[i] != max(y_pred[i])] = 0

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    print('\n', classification_report(y_test, y_pred, digits=4, target_names=target_names))

    return acc[1]


def acc_on_additional_test_set_b2(test_img_path, test_label_path, model_path):
    face_shape_roi, face_shape_onehot = get_test_data_b2(test_img_path, test_label_path)

    cnn_model = load_model(model_path)
    acc_of_cnn = cnn_model.evaluate(face_shape_roi, face_shape_onehot, verbose=1)
    print('B2: Accuracy of 5d-CNN on additional test data is:', acc_of_cnn[1])

    return acc_of_cnn[1]


def B2_network_6_classes():
    inputs = Input(shape=(55, 70, 3))
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    flatten1 = Flatten()(pool2)
    dense1 = Dense(32, activation='relu')(flatten1)
    dense1 = Dropout(0.2)(dense1)
    dense2 = Dense(6, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    return model


# input 5-dimension model and original img/label
def train_cnn_model_b2_6_classes(x_train, y_train):
    print("B2: Training B2 6 classes CNN model...")

    model = B2_network_6_classes()

    # checkpointer = ModelCheckpoint(filepath='B2_eye_color_ROI_cnn.h5', verbose=1, save_best_only=True)
    eye_color_history = model.fit(x_train, y_train, epochs=10,
                                  batch_size=16,
                                  # callbacks=[checkpointer],
                                  verbose=1,
                                  shuffle=True,
                                  validation_split=0.2)

    print("B2: 6d CNN model has been trained successfully")
    return eye_color_history


# input 6-dimension model and image/label of test set
def evaluate_6d_cnn_model_b2(model_6d_path, x_test, y_test):

    model_b2_6classes = load_model(model_6d_path)

    acc = model_b2_6classes.evaluate(x_test, y_test, verbose=1)

    y_pred = model_b2_6classes.predict(x_test)

    print('B2: Accuracy of 6d-CNN on split test data is:', acc[1])

    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] == max(y_pred[i])] = 1
        y_pred[i][y_pred[i] != max(y_pred[i])] = 0

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'sunglasses']
    print(classification_report(y_test, y_pred, digits=4, target_names=target_names))

    return acc[1]


# input 6-dimension model and image/label of test set
def evaluate_6d_cnn_model_test(model_5d_path, model_6d_path, img_path, label_path):

    eye_color_roi, eye_color_6d = add_class_6_to_label(model_5d_path, img_path, label_path)

    model_b2_6classes = load_model(model_6d_path)

    acc = model_b2_6classes.evaluate(eye_color_roi, eye_color_6d, verbose=1)

    y_pred = model_b2_6classes.predict(eye_color_roi)

    print('B2: Accuracy of 6d-CNN on additional test data is:', acc[1])

    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] == max(y_pred[i])] = 1
        y_pred[i][y_pred[i] != max(y_pred[i])] = 0

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'sunglasses']
    print('\n', classification_report(eye_color_6d, y_pred, digits=4, target_names=target_names))

    return acc[1]
