# AMLS-assignment

---
You can get eveything that's needed for the assignment from [here](https://drive.google.com/drive/folders/144gfcSB9e0KBBcgpNtOWTIIfP2pvp6xA?usp=sharing).

Note that github repository **does not** contain all the files.


*All the experiments of this assignment are carried out using jupyter notebook in local laptop. However, in order to keep folders neat and clean, only organized python files with necessary functions are provided here.*


## Running Environment
---
All codes are written in python 3.8.5 environment and experimented in Mac OS. 

Some packetages needed for this task are given here long with its version. 

+ sklearn
+ keras 2.4.3
+ numpy 1.18.5
+ pandas 1.1.3
+ tensorflow 2.3.1
+ face_recognition 1.3.0

A special facial feature extraction library is used here called face_recogniton, you may have to install it using
``` bash 
                                   pip3 install face_recognition
```

## To run 
---
**The images used for each task should be copied into the correct folder under the Dataset folder first.**

If everything is downloaded and installed, you can simply run the main.py to get the results.

An additional main.ipynb is provided with expected output, you can also run this file as each section can be run separately.

## Attentions
Some functions such as model training and grid search functions, which takes a lot of time to process, are commented out in each task section in the default main.py file. You can uncomment them if you wish to see the training process. 

The model trained in main.py won't be saved as the *checkpointer* is commented out in the model_.py in each folder of four tasks. If you want to wait and retrain a model, just delete the # checkpointer in the model_.py in each folder, but remember to change the name and path as it may cover the pretrained model.

The dictionary of parameters used in grid search is smaller than experiment, that is, the number of parameters to be tuned is smaller, because I don't want it spends too much time here in grid search.

*Import warnings* at the beginning of the main.py file is because when running grid search for SVM in task A1, some warnings of 'Precision is ill-defined and being set to 0.0 due to no predicted samples.' happens, which is weird since I have provided features of both two classes to train, and the same grid search method used in task A2 works.

The facial feaure extraction (load data in task A1 and A2) and grid search functions take some time to finish.

Using the main.ipynb file is more convienient to train and stop training a model.

## Project Organization
---

### Base folder
---
* main.py: used to run the whole project.
* main.ipynb: same as main.py, only with expected results and can be run separately for each section.
* Datasets: empty folder which requires to put in four datasets.
* A1 A2 B1 B2 contains preprocessing and model python files as well as pre-trained models used for each task.



### Folder A1
---
* A1_transfer_test_augment.h5: the pre-trained model based on the inception-v3 model used to classify gender.
* inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5: the inception-v3 model used for transfer learning.
* preprocess_a1.py: contains functions related to 
  - load images and labels from directory,
  - find images that can be recognized using face_recognition library,
  - extract facial features using face_recognition library,
  - normalise coordinates within its corresponding location of face.

* models_a1.py: contains 
  - KNN model and grid search for k of KNN,
  - SVM model and grid search for kernels and parameters of SVM,
  - CNN model and it's training and evaluating functions.
 
get_a1_data() and get_a1_data_cnn() of preprocess_a1.py are used in the main.py to load data.

The number of parameters that to be tuned in grid_search_for_svm_a1(x_train, x_test, y_train, cv_folds) of models_a1.py is reduced to save time.


### Folder A2
---
This folder is very similar to A1 except that 
+ lips features instead of the whole facial features are used,
+ CNN model is not used here,
+ a DNN network is proposed in models_a2.py

and following are details:

* A2_smiling_72_face_orig_normalizaed.h5: the pre-trained model based on the extracted features used to classify emotion.
* preprocess_a2.py: contains functions related to 
  - load images and labels from directory,
  - find images that can be recognized using face_recognition library,
  - extract lips features using face_recognition library,
  - normalise coordinates within its corresponding location of face.

* models_a2.py: contains 
  - KNN model and grid search for k of KNN,
  - SVM model and grid search for kernels and parameters of SVM,
  - DNN model and it's training and evaluating functions.
 
get_a2_data() of preprocess_a2.py is used in the main.py to load data.

The number of parameters that are tuned in grid_search_for_svm_a2(x_train, x_test, y_train, cv_folds) of models_a2.py is reduced to save time.



### Folder B1
---
* B1_face_shape_ROI_cnn.h5: pre-trained cnn model used to classify face shapes
* preprocess_b1.py: contains functions related to
  - load images and labels from directory,
  - extract ROI of the lower end of the face from each image,
  - change the labels to one-hot form.
* models_b1.py: contains functions of training and evaluating CNN model.

### Folder B2
---
* B2_eye_color_ROI_cnn.h5: pre-trained 5d-cnn model used to classify five eye colors
* B2_eye_color_ROI_cnn_class6.h5: pre-trained 6d-cnn model used to classify six eye colors including sunglasses
* preprocess_b2.py: contains functions related to
  - load images and labels from directory,
  - extract ROI of left eye from each image,
  - change the labels to one-hot form,
  - **relabel images with sunglasses to a new class**
* models_b2.py: contains functions of training and evaluating both 5d-CNN model and 6d-CNN model.











