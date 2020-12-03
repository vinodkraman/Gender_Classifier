# Matroid ML Challenge

Hello there! Here, you can find all the relevant information for Tasks 1 and 2 of the Matroid ML Challenge. All code was written in Google Collab due to the lack of disk space on my local computer. For completeness sake I have downloaded these notebooks as .py and files and included them in the tarball. That said, I highly suggest using the links in the sections below to access the Google Collab notebook directly as this will save much time with any required dependencies. 

Note: although GPUs are available in Google Collab, per the instructions, I did not enable them during runtime. 

## Relevant Links and Resources
- Caffe-to-Keras Convertor: <https://github.com/pierluigiferrari/caffe_weight_converter>
- Model Conversion: <https://colab.research.google.com/drive/1N13QnNBCiaAB-A3FOn8irMoRb44mduc5?usp=sharing>
- Transfer Learning/Gender Classification: <https://colab.research.google.com/drive/1kkXN3EMy3tU1WnXf2HEQlqdFNzXwk7IL?usp=sharing>


## Task 1: Model Conversion

To convert the VGG Face Descriptor to Tensorflow format, I used an open-source caffe-to-keras weight conversion [tool](https://github.com/pierluigiferrari/caffe_weight_converter) to extract the weights from the `VGG_FACE.caffemodel` to a .h5 file compatible with Keras. To convert a `.caffemodel` file to a Keras-compatible HDF5 file with verbose console output, run the following on the command line:
```c
python caffe_weight_converter.py 'desired/name/of/your/output/file/without/file/extension' \
                                 'path/to/the/caffe/model/VGG_FACE_deploy_full.prototxt' \
                                 'path/to/the/caffe/VGG_FACE.caffemodel' \
                                 --verbose
```

The `VGG_FACE_deploy_full.prototxt` can be found in the tarball. Note, that the `VGG_FACE_deploy_full.prototxt` is not the same prototxt file as what was provided by the paper. 

Once the weights have been extracted to a .h5 file compatible with Keras, it can be easily loaded into a high-level tf.Keras model with the same architecture as the VGG Face Descriptor. The Tensorflow version of the fully converted VGG Face Descriptor can be found [here](https://colab.research.google.com/drive/1N13QnNBCiaAB-A3FOn8irMoRb44mduc5?usp=sharing). The .py file of the notebook is also included in the tarball as `model_conversion.py` 

## Task 2: Transfer Learning/Gender Classification

All commented code for this task can be found [here](https://colab.research.google.com/drive/1kkXN3EMy3tU1WnXf2HEQlqdFNzXwk7IL?usp=sharing) Its corresponding .py file is included in the tarball for completeness sake as `transfer_learning_gender_classification.py`

The easiest way to run the code is by running each cell of the linked Google Collab notebook after uploading the model weights included (more info included in notebook).
### Data Preprocessing
The dataset was downloaded and unzipped from the following .targz link <https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.>

The dataset is composed of two-high level directories named `aligned` and `valid`. Within each of these directories exist 140 age-gender folders, corresponding to 70 age groups (1-70) for both men and women. Within each age-gender folder contains the images for that gender and that age. 

Although it would have been ideal to construct training, validation, and testing sets by splitting across combined images of the valid and aligned directories, this turned out to make training of the overall model (frozen pre-trained VGG_FACE + classification head) very computationally expensive due to the sheer size of the dataset (this also turned out to be problematic when only considering the images in only one of the valid or aligned directories). This is because even with the weights of a pre-trained VGG_FACE frozen, inference through the pre-trained VGG_FACE turned out to be quite slow (10s for batch of 20 images), probably due to its massive architecture.  

To combat this issue, I manually created a more computationally feasible dataset via structured subsampling. In an attempt to somewhat preserve the distribution over gender and age examples in the original dataset, I constructed the smaller more feasible dataset by combining a random sample of around 20 images from each age-gender folder of the aligned directory. Given that there are 140 age-gender folders in the aligned directory, the size of the more feasible dataset is 70 * 20 * 2 = 2800 images.

Once the feasible dataset was constructed, its rows were shuffled, and split 60/20/20 into training, validation, and testing datasets respectively. This split was chosen because our feasible dataset is of smaller size. Once the feasible dataset was split, the images were normalized by computing the mean image over the training set, and then subtracting this image from the training, validation, and testing sets. Note this is the same procedure done by Parkhi et al. and the assumption is that the gender dataset and the dataset used by Parkhi et al. are similar. Note that Parkhi et al. did not mention anything about rescaling the pixel values to [0,1], and therefore I did not perform this rescaling either. 

Once all three datasets have been normalized, they are then ported over into tf.keras.Dataset by using:

```python 
train_dataset = tf.data.Dataset.from_tensor_slices((norm_images_train, labels_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((norm_images_valid, labels_valid))
test_dataset = tf.data.Dataset.from_tensor_slices((norm_images_test, labels_test))

```


### Model Building
To make use of the pre-trained VGG Face Descriptor for transfer learning, I first extract weights of the VGG Face Descriptor **without** the fully-connected classification layers into a .h5 file. This is performed just like above, except now using a modified `VGG_FACE_deploy.prototxt` file that discards the fully-connected classification layers. This modified `VGG_FACE_deploy.prototxt` file is included in the tarball as `VGG_FACE_deploy_truncated.prototxt`. With the weights extracted for the truncated model, we can rebuild its truncated architecture in Tensorflow using the high-level `tf.keras.Sequential ` module and simply load in the weights like in Task 1.  This will serve as our base model/feature extractor for transfer learning.

Once the convolution base model is created, I freeze its weights and stack a classification head on top. The classification head consists of a 2D Global Average Pooling, a 32-unit Fully-Connected layer, and the final prediction neuron. Dropout was also used after the 32-unit Fully-Connected layer to reduce overfitting. 

 The complete code for building the base model and the overall model is included and well documented in the Google Collab linked above. 

### Training
Once the overall model architecture is complete, we compile it by calling

```python
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

The base learning rate was chosen after hyperparameter tuning. Once the model has been compiled, it is trained by running 

```python
epochs=10
history = new_model.fit(
  train_dataset,
  validation_data=valid_dataset,
  epochs=epochs
)
```

 The model was trained for only 10 epochs due to the computational constraints. The weights of the trained model and its architecture are saved using the SavedModel format and included in the tarball as `weights_arch_transfer_model`.  Note, I also saved the weights and architecture using the HDF5 standard and can be found in the tarball as `weights_arch_transfer_model_h5.h5`. 

 The complete code for training and saving the trained model is included and documented in the Google Collab linked above. 

### Evaluation
With a trained model, we can evalaute the test set compiled during the data preprocessing step. This can be done by calling `model.predict()` on the test dataset to recieve predictions and then using `tf.keras.metrics` on the predictions with the true labels to compute a given metric. We consider 4 main metrics for binary classification: 

- Accuracy
- Precision
- Recall
- Specificity
- AUC

We compute these metrics for the overall test set, only the males in the test set, and only the females in the test set. The results are summarized below and also included in the Evaluation section of the linked Google Collab for reference. 

### Results
We summarize the results/metrics of our trained model evaluated on the test dataset below.

Metric | Test Overall | Test Male (label = 0) | Test Female (label = 1)
--- | --- | --- | --- |--- |---| 
Accuracy | 0.927 | 0.922 | 0.932 | 
Precision | 0.915 | 0.0 | 1.0 | 
Recall | 0.932 | 0.0 | 0.932 | 
Specificity| 0.922 | 0.922 | -
AUC | 0.927 | 0.0 | 0.0 | 


 I believe that amongst other possible issues such as overfitting, the somewhat poor performance of the trained model can be mainly attributed to the computational constraints placed on the number of epochs and size of the training database. In addition, the assumption on the similarity of the gender dataset and the dataset Parkhi et al. used may not actually hold, and thus the standardization procedure used on the input images to the transfer-learned model may be invalid.
