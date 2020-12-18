# add file path of each task to system path
import sys
sys.path += ['./A1/', './A2', './B1', './B2']
from models_a1 import *
from preprocess_a1 import get_a1_data, get_a1_data_cnn
from models_a2 import *
from preprocess_a2 import get_a2_data
from models_b1 import *
from preprocess_b1 import get_b1_data
from models_b2 import *
from preprocess_b2 import get_b2_data, add_class_6_to_label

import warnings
warnings.filterwarnings("ignore")

# define image, label and model path
img_path_A = './Datasets/celeba/img/'
label_path_A = './Datasets/celeba/labels.csv'
test_img_path_A = './Datasets/celeba_test/img/'
test_label_path_A = './Datasets/celeba_test/labels.csv'
a1_saved_model_path = './A1/A1_transfer_test_augment.h5'
a2_saved_model_path = './A2/A2_smiling_72_face_orig_normalizaed.h5'

img_path_B = './Datasets/cartoon_set/img/'
label_path_B = './Datasets/cartoon_set/labels.csv'
test_img_path_B = './Datasets/cartoon_set_test/img/'
test_label_path_B = './Datasets/cartoon_set_test/labels.csv'
b1_saved_model_path = './B1/B1_face_shape_ROI_cnn.h5'
b2_saved_model_path = './B2/B2_eye_color_ROI_cnn.h5'
b2_saved_6_classes_model_path = './B2/B2_eye_color_ROI_cnn_class6.h5'


# ======================================================================================================================


# Task A1
# load data for KNN and SVM, this takes some time
feature_train, feature_test, feature_train_label, feature_test_label = get_a1_data(img_path_A, label_path_A)

# accuracy of knn and svm on split test set
a1_knn_acc, knn_clf_a1 = knn_model_a1(feature_train, feature_test, feature_train_label, feature_test_label, k=4)
a1_svm_acc, svm_clf_a1 = svm_model_a1(feature_train, feature_test, feature_train_label, feature_test_label)

# uncomment below two lines to see grid search process for knn and svm, it may take several minutes to finish
# knn_scores_a1 = grid_search_for_knn_a1(feature_train, feature_test, feature_train_label, feature_test_label, search_num=30)
# best_pred_a1 = grid_search_for_svm_a1(feature_train, feature_test, feature_train_label, cv_folds=5)

# load data for CNN
x_train, x_test, x_valid, y_train, y_test, y_valid = get_a1_data_cnn(img_path_A, label_path_A)

# uncomment this line to train cnn model, checkpointer is commented out in case the pretrained model get covered
# hist_a1 = train_cnn_model_a1(x_train, y_train, x_valid, y_valid)

# accuracy of cnn on split test set
a1_cnn_acc = evaluate_cnn_model_a1(a1_saved_model_path, x_test, y_test)

# acc of three methods on additional test set
acc_of_knn_a1, acc_of_svm_a1, acc_of_cnn_a1 = acc_on_additional_test_set_a1(test_img_path_A, test_label_path_A, a1_saved_model_path, knn_clf_a1, svm_clf_a1)


# ======================================================================================================================


# Task A2
# load data for KNN and SVM
smile_train, smile_test, smile_train_label, smile_test_label = get_a2_data(img_path_A, label_path_A)

# accuracy of knn and svm on split test set
a2_knn_acc, knn_clf_a2 = knn_model_a2(smile_train, smile_test, smile_train_label, smile_test_label, k=11)
a2_svm_acc, svm_clf_a2 = svm_model_a2(smile_train, smile_test, smile_train_label, smile_test_label)
a2_acc_dnn = evaluate_dnn_model_a2(a2_saved_model_path, smile_test, smile_test_label)

# uncomment following two lines to see grid search process for knn and svm
# knn_scores_a2 = grid_search_for_knn_a2(smile_train, smile_test, smile_train_label, smile_test_label, search_num=50)
# best_pred_a2 = grid_search_for_svm_a2(smile_train, smile_test, smile_train_label, cv_folds=5)

# uncomment this line to train cnn model, checkpointer is commented out in case the pretrained model get covered
# hist_a2 = train_dnn_model_a2(smile_train, smile_train_label)

# acc of three methods on additional test set
acc_of_knn_a2, acc_of_svm_a2, acc_of_dnn_a2 = \
    acc_on_additional_test_set_a2(test_img_path_A, test_label_path_A, a2_saved_model_path, knn_clf_a2, svm_clf_a2)


# ======================================================================================================================


# Task B1
# load normalised data
shape_train, shape_test, shape_train_label, shape_test_label = get_b1_data(img_path_B, label_path_B)

# uncomment this line to train the model
# face_shape_history = train_cnn_model_b1(shape_train, shape_train_label)

b1_acc = evaluate_cnn_model_b1(b1_saved_model_path, shape_test, shape_test_label)

# acc on additional test set
acc_of_cnn_b1 = acc_on_additional_test_set_b1(test_img_path_B, test_label_path_B, b1_saved_model_path)


# ======================================================================================================================


# Task B2
# load normalised data
eye_train, eye_test, eye_train_label, eye_test_label = get_b2_data(img_path_B, label_path_B)

# uncomment this line to train the 5d-cnn model
# eye_color_history = train_cnn_model_b2(eye_train, eye_train_label)

# acc of 5d-cnn model on test set
b2_acc = evaluate_cnn_model_b2(b2_saved_model_path, eye_test, eye_test_label)
# acc of 5d-cnn model on additional test set
acc_of_cnn_b2 = acc_on_additional_test_set_b2(test_img_path_B, test_label_path_B, b2_saved_model_path)

# add sunglasses as a new label
eye_color_roi, eye_color_6d_label = add_class_6_to_label(b2_saved_model_path, img_path_B, label_path_B)

# uncomment this line to train the 6d-cnn model
# eye_color_6d_hist = train_cnn_model_b2_6_classes(eye_color_roi, eye_color_6d_label)

# acc of 6d-cnn model on test set
acc_original = evaluate_6d_cnn_model_b2(b2_saved_6_classes_model_path, eye_color_roi, eye_color_6d_label)
# acc of 6d-cnn model on additional test set
acc_test = evaluate_6d_cnn_model_test(b2_saved_model_path, b2_saved_6_classes_model_path,
                                          test_img_path_B, test_label_path_B)



# ======================================================================================================================

print('TA1(test_set)       :KNN:{}, SVM:{}, CNN:{};\nTA1(additional_set) :KNN:{}, SVM:{}, CNN:{};\n\n'
      'TA2(test_set)       :KNN:{}, SVM:{}, DNN:{};\nTA2(additional_set) :KNN:{},  SVM:{}, DNN:{};\n\n'
      'TA3(test_set)       :CNN:{};\nTA3(additional_set) :CNN:{};\n\n'
      'TA4(test_set)       :5d-CNN:{},  6d-CNN:{};\nTA4(additional_set) :5d-CNN:{},  6d-CNN:{};\n'
      .format(a1_knn_acc, a1_svm_acc, a1_cnn_acc,
              acc_of_knn_a1, acc_of_svm_a1, acc_of_cnn_a1,
              a2_knn_acc, a2_svm_acc, a2_acc_dnn,
              acc_of_knn_a2, acc_of_svm_a2, acc_of_dnn_a2,
              b1_acc, acc_of_cnn_b1,
              b2_acc, acc_original, acc_of_cnn_b2, acc_test
             ))
