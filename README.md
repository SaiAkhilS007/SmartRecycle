# SmartRecycle 

## Introduction
The lack of accessible tools for segregating waste and finding immediate recycling or reuse solutions leaves individuals unaware of simple steps they can take. Additionally, locating nearby drop-off points for waste types like organic, e-waste, and medical waste remains challenging. The absence of a platform that connects users with disposal facilities and provides instant reuse or recycling ideas inspired us to create a solution for better waste management practices.
### Objective
- Use Traditional ML models, Ensemble models and Deep Learning Models to classify waste images into categories like plastic, metal, or organic categories.
- Provide recycling suggestions, reuse ideas, and safe disposal methods for hazardous or non-recyclable waste.
- Suggest nearby recycling centers or drop-off points using geolocation and mapping tools.
## Research Questions
- Research Question 1: How can image classification models be optimized to accurately categorize waste into distinct categories for better waste management 
  practices?
- Research Question 2: How can location-based services and predictive models be integrated to suggest optimal waste disposal options based on user location and 
  waste type?
## Data Source
Data set has been downloaded from this URL titled "ImageNet" : https://www.image-net.org/index.php
- **Main categories chosen(Images)**
  1) *Wood*
  2) *cardboard*
  3) *e-waste*
  4) *glass*
  5) *medical*
  6) *metal*
  7) *paper*
  8) *plastic*
## Data Augmentation
- Performed data augmentation techniques on categories like Cardboard,medical,organic_waste,textiles,Wood for balancing the datasets
Image Counts by Category:
- cardboard: Main = 2332, Augmented = 410, Total = 2742
- plastic: Main = 2617, Augmented = 0, Total = 2617
- glass: Main = 2518, Augmented = 0, Total = 2518
- medical: Main = 1605, Augmented = 1088, Total = 2693
- paper: Main = 2749, Augmented = 0, Total = 2749
- e-waste: Main = 2405, Augmented = 0, Total = 2405
- organic_waste: Main = 277, Augmented = 2209, Total = 2486
- textiles: Main = 335, Augmented = 2155, Total = 2490
- metal: Main = 2259, Augmented = 0, Total = 2259
- Wood: Main = 347, Augmented = 2111, Total = 2458
  

## Model Performance: Traditional Machine Learning Models

| **Model Used**       | **Features Extracted Using** | **Train Accuracy** | **Test Accuracy** | **F1 Score** | **Precision** | **Recall** |
|-----------------------|------------------------------|---------------------|--------------------|--------------|---------------|------------|
| Logistic Regression   | InceptionV3                | 100%               | 90%                | 0.90         | 0.90          | 0.90       |
| Logistic Regression   | DenseNet121                | 93%                | 85%                | 0.85         | 0.85          | 0.85       |
| Decision Tree         | InceptionV3                | 65%                | 62%                | 0.69         | 0.62          | 0.64       |
| Decision Tree         | DenseNet121                | 99%                | 31%                | 0.33         | 0.31          | 0.31       |
| SVM                   | InceptionV3                | 100%               | 90%                | 0.90         | 0.90          | 0.90       |
| SVM                   | DenseNet121                | 92%                | 91%                | 0.91         | 0.91          | 0.91       |
 
  
## Ensemble models Tackled

  <img src="Visualizations/ensemble.jpg" width ="500" height = "400" />

## Model Performance: Ensemble Models

| **Model Used**       | **Features Extracted Using** | **Train Accuracy** | **Test Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|------------------------------|---------------------|--------------------|---------------|------------|--------------|
| Random Forest         | InceptionV3                | 91%                | 82%                | 0.84          | 0.82       | 0.82         |
| Random Forest         | DenseNet121                | 99%                | 72%                | 0.72          | 0.70       | 0.70         |
| XGBoost               | InceptionV3                | 100%               | 90%                | 0.90          | 0.90       | 0.90         |
| XGBoost               | DenseNet121                | 95%                | 87%                | 0.88          | 0.88       | 0.88         |
| AdaBoost              | InceptionV3                | 63%                | 62%                | 0.62          | 0.62       | 0.62         |
| AdaBoost              | DenseNet121                | 69%                | 67%                | 0.68          | 0.68       | 0.67         |


## Model Performance: Hybrid Models (SVM + XGBoost)

| **Model Used**       | **Features Extracted Using** | **Train Accuracy** | **Test Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|------------------------------|---------------------|--------------------|---------------|------------|--------------|
| SVM + XGBoost         | InceptionV3                | 99%                | 92%                | 0.92          | 0.92       | 0.92         |
| SVM + XGBoost         | DenseNet121                | 93.3%              | 92%                | 0.93          | 0.93       | 0.93         |


## Model Performance and Architecture Overview: Deep Learning Models

| **Model Used**       | **Overview of the Model**                                                                                                                                                                                                                       | **Train Accuracy** | **Test Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|--------------------|---------------|------------|--------------|
| **DenseNet**          | Input Layer: Receives images of size (244, 244, 3). Hidden Layers: The output of the DenseNet201 convolutional layers is passed through a GlobalAveragePooling2D layer to reduce spatial dimensions (224x224x3 → 1x1x1024). Then, it goes through a Dropout layer (rate 0.3) to prevent overfitting, followed by a Dense layer (num_classes neurons → softmax activation). Output Layer: Outputs a probability distribution with num_classes neurons using softmax activation for multi-class classification. | 89%                | 89%               | 0.89          | 0.89       | 0.89         |
| **Inception V3**      | Input Layer: The model receives images of size (299, 299, 3). Hidden Layers: The output from the InceptionV3 convolutional layers is passed through a GlobalAveragePooling2D layer to reduce spatial dimensions (from 299x299xchannels to a 1D vector). Next, a Dropout layer with a rate of 0.3 is applied to minimize overfitting, followed by a Dense layer with num_classes neurons and a softmax activation function. Output Layer: Produces a probability distribution across num_classes categories, using softmax activation for multi-class classification. | 95%                | 90%               | 0.95          | 0.95       | 0.95         |
| **ResNet**            | Input Layer: Receives images of size (224, 224, 3) and processes them through a series of residual blocks with skip connections to maintain information flow and support deeper learning. Feature Extraction: Uses a GlobalAveragePooling2D layer to compress spatial dimensions (e.g., 7x7x2048 → 1x1x2048) and a Dropout layer (rate 0.3) to minimize overfitting. Output Layer: A final Dense layer with num_classes neurons applies softmax activation to produce probabilities for multi-class classification, selecting the most likely waste category. | 99%                | 94%               | 0.94          | 0.94       | 0.94         |


From the above table we can conclude ResNet achieved the highest test accuracy (94%) and consistent metrics (Precision, Recall, and F1 Score: 0.94), making it the best overall model.



## Research Question 1: How can image classification models be optimized to accurately categorize waste into distinct categories for better waste management practices?

The deep learning models, however, outperformed traditional and ensemble approaches. ResNet achieved the highest test accuracy of 94% with balanced metrics (Precision, Recall, and F1 Score: 0.94), making it the most reliable model. While InceptionV3 and DenseNet showed strong performance.




Top 3 YouTube recommendations for reusing Plastic (Cosine Similarity):
Title: DIY TOY Super creative smart recycling ideas/waste plastic diy craft ideas/plastic hacks/tiktokviral, Cosine Similarity: 0.2424, URL: https://www.youtube.com/watch?v=uYYZTMjE9pA
Title: Clever ways to reuse and recycle empty plastic bottles, Cosine Similarity: 0.2009, URL: https://www.youtube.com/watch?v=O7JkFJXcOKM
Title: 4 Brilliant Ideas From Plastic Cans! Don&#39;t Throw Away Empty Cans!!!, Cosine Similarity: 0.1833, URL: https://www.youtube.com/watch?v=sGl1eihw4cE

Top 3 YouTube recommendations for reusing Plastic (Euclidean Similarity):
Title: DIY TOY Super creative smart recycling ideas/waste plastic diy craft ideas/plastic hacks/tiktokviral, Euclidean Similarity: 0.4482, URL: https://www.youtube.com/watch?v=uYYZTMjE9pA
Title: Clever ways to reuse and recycle empty plastic bottles, Euclidean Similarity: 0.4417, URL: https://www.youtube.com/watch?v=O7JkFJXcOKM
Title: 4 Brilliant Ideas From Plastic Cans! Don&#39;t Throw Away Empty Cans!!!, Euclidean Similarity: 0.4390, URL: https://www.youtube.com/watch?v=sGl1eihw4cE

