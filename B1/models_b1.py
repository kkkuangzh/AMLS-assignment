from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from preprocess_b1 import get_test_data_b1


def B1_network():
    inputs = Input(shape=(100, 200, 3))
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
def train_cnn_model_b1(x_train, y_train):
    print("Training B1 cnn model...")
    model = B1_network()
    # checkpointer = ModelCheckpoint(filepath='B1_face_shape_ROI_cnn.h5', verbose=1, save_best_only=True)
    face_shape_history = model.fit(x_train, y_train, epochs=5,
                                   batch_size=16,
                                   # callbacks=[checkpointer],
                                   verbose=1,
                                   shuffle=True,
                                   validation_split=0.2)

    print("Model has been trained successfully")
    return face_shape_history


def evaluate_cnn_model_b1(model_path, x_test, y_test):

    model = load_model(model_path)
    acc = model.evaluate(x_test, y_test, verbose=1)

    y_pred = model.predict(x_test)

    print('B1: Accuracy of CNN on split test set is:', acc[1])
    for i in range(len(y_pred)):
        y_pred[i][y_pred[i] == max(y_pred[i])] = 1
        y_pred[i][y_pred[i] != max(y_pred[i])] = 0

    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    print('\n', classification_report(y_test, y_pred, digits=4, target_names=target_names))

    return acc[1]


def acc_on_additional_test_set_b1(test_img_path, test_label_path, model_path):
    face_shape_roi, face_shape_onehot = get_test_data_b1(test_img_path, test_label_path)

    cnn_model = load_model(model_path)
    acc_of_cnn = cnn_model.evaluate(face_shape_roi, face_shape_onehot, verbose=1)
    print('B1: Accuracy of CNN on additional test set is:', acc_of_cnn[1])

    return acc_of_cnn[1]



