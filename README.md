# Convolutional Neural Network for Alzheimer's Detection From MRI Scans
![example_mri ](https://github.com/user-attachments/assets/d886fa1e-39cf-4d58-8f13-048e2ea6073b)


## Background
Alzheimer’s disease (AD) is a neurological disorder that results in diminished cognitive function. AD onset most often occurs when people are in their mid 60s and is the most frequent cause of dementia in seniors. It is currently estimated that over 6 million American’s over 65 have AD [5]. There are currently no cures for AD, however there are some treatment strategies. Early detection and intervention have been shown to slow disease progression and improve the quality of life for individuals suffering from AD [4]. Definitively diagnosing AD while someone is alive remains a challenge for the medical community and several metrics need to be assessed to determine if an individual is suffering from AD [3]. These methods may include brain scans such as magnetic resonance imaging (MRI), cognitive assessments through testing of memory, attention and problem solving, overall health assessment, and examining environmental and biological factors [3]. Developing models that help detect early-stage AD would be a great help to those suffering from the disease. The focus of this project will be to build a convolutional neural network (CNN) to detect AD in MRI scans.

## DATA 

The dataset used for this project consists of images from MRI brain scans. The MRI technique is non-invasive and can produce detailed images of soft tissue such as brain tissue [1]. Changes in brain structure such as cerebral atrophy (shrinking of the brain), and abnormal protein build up are characteristics of AD [2]. The data is comprised of four classes, Non-Demented, Mild Demented, Moderate Demented, and Very Mild Demented.

## MODELS 

CNNs will be used to detect AD in brain tissue by analyzing MRI images. CNNs are an appropriate choice for this task for several reasons. For one they use convolutional layers with local receptive fields to recognize patterns such as edges, textures and shape. Given the changes in brain structure associated with AD [2], detecting patterns such as these could be helpful in diagnosing the disease. CNNs process images through multiple layers, they learn to extract increasingly complex features. Early layers detect simple structures, while deeper layers can capture more abstract patterns, such as those associated with cerebral atrophy and protein build-up.

## Results
<img width="1041" alt="first_model_results" src="https://github.com/user-attachments/assets/628c45f9-1916-4399-a4e9-d5d4d56cd62e" />

REFERECNCES:

[1] Ashby, K., Adams, B. N., & Shetty, M. (2022, November 14). Appropriate magnetic resonance imaging ordering. StatPearls - NCBI Bookshelf. https://www.ncbi.nlm.nih.gov/books/NBK565857/ [2] Coupé, P., Manjón, J. V., Lanuza, E., & Catheline, G. (2019). Lifespan changes of the human brain in Alzheimer’s disease. Scientific Reports, 9(1). https://doi.org/10.1038/s41598-019-39809-8 [3] “How Is Alzheimer’s Disease Diagnosed?”. National Institute on Aging. Dec.08, 2022. https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed [4] Rasmussen, J., & Langerman, H. (2019). Alzheimer’s Disease – Why We Need Early Diagnosis. Degenerative Neurological and Neuromuscular Disease, Volume 9, 123–130. https://doi.org/10.2147/dnnd.s228939 [5] “What Is Alzheimer’s Disease?”. National Institute on Aging, Jul. 08, 2021. https://www.nia.nih.gov/health/alzheimers-and-dementia/what-alzheimers-disease

Alzheimer MRI Preprocessed Dataset Available at Kaggle: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-
