# Convolutional Neural Network for Alzheimer's Detection From MRI Scans
![example_mri ](https://github.com/user-attachments/assets/d886fa1e-39cf-4d58-8f13-048e2ea6073b)


## Background
Alzheimer’s disease (AD) is a neurological disorder that results in diminished cognitive function. AD onset most often occurs when people are in their mid 60s and is the most frequent cause of dementia in seniors. It is currently estimated that over 6 million American’s over 65 have AD [5]. There are currently no cures for AD, however there are some treatment strategies. Early detection and intervention have been shown to slow disease progression and improve the quality of life for individuals suffering from AD [4]. Definitively diagnosing AD while someone is alive remains a challenge for the medical community and several metrics need to be assessed to determine if an individual is suffering from AD [3]. These methods may include brain scans such as magnetic resonance imaging (MRI), cognitive assessments through testing of memory, attention and problem solving, overall health assessment, and examining environmental and biological factors [3]. Developing models that help detect early-stage AD would be a great help to those suffering from the disease. The focus of this project will be to build a convolutional neural network (CNN) to detect AD in MRI scans.

## Data & Methodology

### DATA 

The dataset used for this project consists of images from MRI brain scans. The MRI technique is non-invasive and can produce detailed images of soft tissue such as brain tissue [1]. Changes in brain structure such as cerebral atrophy (shrinking of the brain), and abnormal protein build up are characteristics of AD [2]. The data is comprised of four classes, Non-Demented, Mild Demented, Moderate Demented, and Very Mild Demented.

- The data used for this work is available at Kaggle, https://www.kaggle.com/datasets/ninadaithal/imagesoasis
  - Acknowledgments: “Data were provided 1-12 by OASIS-1: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382”
  - Citation: OASIS-1: Cross-Sectional: https://doi.org/10.1162/jocn.2007.19.9.1498
- **The data is not balanced**, total images: 86437 with
  - Non Demented images: 67222
  - Very mild Dementia images: 13725
  - Moderate Dementia images: 488
  - Mild Dementia images: 5002
 
### Languge: Python
  - [TensorFlow](https://www.tensorflow.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [Matplotlib](https://matplotlib.org/)

### Hardware: 
  - Dual NVIDIA T4 GPUs T4 x2 accelerator (ran on Kaggle's platform)

### CNN Architecture

CNNs are an appropriate choice for this task for several reasons. For one they use convolutional layers with local receptive fields to recognize patterns such as edges, textures and shape. Given the changes in brain structure associated with AD [2], detecting patterns such as these could be helpful in diagnosing the disease. CNNs process images through multiple layers, they learn to extract increasingly complex features. Early layers detect simple structures, while deeper layers can capture more abstract patterns, such as those associated with cerebral atrophy and protein build-up.

- **Convolutional Layers:** Four convolutional blocks with increasing depth: 64 → 128 → 256 → 512 filters.

- **Data Augmentation:**
  - Added before the first convolutional layer to generate diverse training samples.
  - Randomly rotates, flips, zooms or adjusts the contrast of some images.

- **Regularization and Normalization:**
  - Dropout after each block (progressively increasing)
  - BatchNormalization throughout the network
  - L2 kernel regularization to reduce overfitting
  
- **Pooling:**
  - MaxPooling2D after each convolutional block to reduce spatial dimensions
  - GlobalAveragePooling2D before fully connected layer
  
- **Fully Connected Layer:**
  - Dense layer with 256 units, followed by Dropout and BatchNormalization
  - Final output layer: 4 unit with sofmax activation for multiclass (4) classification

### Training Strategy

- **Optimizer:** Adam with a learning rate initialized at 0.0002

- **Loss Function:** categorical_crossentropy

- **Learning Rate Scheduler:** A scheduler was used ([ReduceLROnPlateau]( https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)) to reduce the learning rate during training to help with model stability.

## Results
The two minority classes, Moderate Dementia and Very mild Dementia are not being identified.
<img width="754" alt="ThirdModelPerformance" src="https://github.com/user-attachments/assets/02c5452c-bf4c-4263-ae79-bbcf549a0a63" />
<img width="637" alt="ThirdModelConfusion" src="https://github.com/user-attachments/assets/05e0bc9f-afd6-4480-8528-1d3f4252238f" />


## To Do
- [ ] Develop a strategy to handle data imbalance.
  - [x] Class weighting:
    Two strategies were attempted, one setting class_weight='balanced', and the other using a customized class weight calculation. Both approaches resulted in all images being predicted as Non Demented. Notebook version: cnn-alzheimer-mri-detection20250526
  - [ ] Creating a custom oversampling pipeline with augmentation.
  - [ ] Using a pretrained architecture such as [MobileNetV2](https://keras.io/api/applications/mobilenet/)
  
        - **Review** Foster, L. (2023, April 18). Identifying Alzheimer’s Disease with Deep Learning: A Transfer Learning Approach. Medium. https://medium.com/@lfoster49203/identifying-alzheimers-disease-with-deep-learning-a-transfer-learning-approach-620abf802631
  - [ ] **Ambitious:** Build a GAN to generate synthetic minority class images 

## REFERECNCES:

[1] Ashby, K., Adams, B. N., & Shetty, M. (2022, November 14). Appropriate magnetic resonance imaging ordering. StatPearls - NCBI Bookshelf. https://www.ncbi.nlm.nih.gov/books/NBK565857/ 

[2] Coupé, P., Manjón, J. V., Lanuza, E., & Catheline, G. (2019). Lifespan changes of the human brain in Alzheimer’s disease. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-39809-8 

[3] “How Is Alzheimer’s Disease Diagnosed?”. National Institute on Aging. Dec.08, 2022. https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed 

[4] Rasmussen, J., & Langerman, H. (2019). Alzheimer’s Disease – Why We Need Early Diagnosis. Degenerative Neurological and Neuromuscular Disease, Volume 9, 123–130. https://doi.org/10.2147/dnnd.s228939 

[5] “What Is Alzheimer’s Disease?”. National Institute on Aging, Jul. 08, 2021. https://www.nia.nih.gov/health/alzheimers-and-dementia/what-alzheimers-disease

