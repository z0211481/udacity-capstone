# Azure Machine Learning Engineer - Capstone Project

This project is the capstone project of Udacity's *Machine Learning Engineer with Microsoft Azure* nanodegree. 
The scope of the project is to present the knowledge obtained from finishing the courses.

## Project Set Up and Installation

Setting up the project in Azure ML:
1. Create new or access a workspace in Azure with your subscription
2. Open Azure Machine Learning Studio
3. Create a Compute instance so you can run your jupyter notebooks
4. Starter files were provided by Udacity, you can find them in [this repository](https://github.com/udacity/nd00333-capstone/tree/master/starter_file)

## Dataset

### Overview

The dataset contains information about visual characteristics of cancerous cells in the breast and ground truth about the diagnosis made based on the cell data. The visuals were measured based on X-rays.

[*Source*](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

### Task

Based on the data we can create a classification model, so if we input the visual characteristics, a diagnosis can be made, whether the cell is benign or malignant.

### Access
For the Automated ML I have uploaded the dataset into the Azure ML Studio as a dataset. For the hyperdrive training I have uploaded the file and read it with pandas.

## Automated ML
For the Automated Machine Learning, I have used the following settings and configuration.

Settings:
- *experiment_timeout_minutes*: 20 (stop after 20 mins of inactivity)
- *max_concurrent_iterations*: 5 (maximum 5 different models run at the same time)
- *primary_metric*: AUC_weighted (the metric based on which the AutoML will choose which model performs the best)
 
Configuration:
- *task*: Classification
- *label_column_name*: diagnosis (this is the column in the dataset that needs to be classified)
- *enable_early_stopping*: True (stop if the models do not improve)

### Results
The best model trained with AutoML is a VotingEnsemble algorithm that combines multiple algorithms and makes a decision based on all of the predictions. The combined algoithms: XGBoostClassifier, LightGBM, SVM, LogisticRegression. The endmodel performed with an accuracy of 0.9824, and a weighted AUC of 0.9970. 
The latter is a pretty high number it would probably be hard to achieve higher weighted AUC. It would be interesting to optimize the models to accuracy instead of weighted AUC.

![](/screenshots/01a_automl.PNG)

![](/screenshots/01b_automl.PNG)

![](/screenshots/02_best_model_automl.PNG)

## Hyperparameter Tuning

For the hyperparameter tuning, I chose a simple K nearest neighbor model, because the data is not complex, an I wanted to see, how a simple model would perform on them.
This model is also not time-consuming.

The hyperparameters to be tuned:
- **n**: number of neighbors in the algorithm
- **weights**: function to be used when calculating the weight of neighboring data points (can be *uniform* or *distance*)
- **p**: power parameter fro the Minkowski metric

#### Sampling: RandomParameterSampling
Random Parameter Sampling chooses parameters from a prespecified set of discrete parameters or a continuous limited set. This sampler chooses parameters randomly, this way we do not have to check each parameter combination. This is a time-efficient way of sampling parameters.


#### Stopping policy: Bandit
The bandit policy terminates runs where the primary metric is not within the specified slack factor (0.1) compared to the best performing model. Setting this policy ensures that models performing 10% worse than already trained models, will not be trained full, therefore we can spare time.

### Results

The best model during the hyperparameter tuning performed with an accuracy of 0.8412. 
You can see the chosen parameters in the following image.
Better accuracy can probably be achieved, if more parameters values are provided and more different models are trained.
In this case, optimization was made based on accuracy. Therefore this is not a fair comparison to the automated ML trained models, which were optimized for weighted AUC.

![](/screenshots/04_best_model_hyperdrive.PNG)

![](/screenshots/03a_hyperdrive.PNG)

![](/screenshots/03b_hyperdrive.PNG)

![](/screenshots/03c_hyperdrive.PNG)

## Model Deployment

Since the model that was trained with Automated ML performed significantly better, I have deployed that one. 



![](/screenshots/05_deployed_model.PNG)

You can query this from Python. Example:

![](/screenshots/06_testing.PNG)

Sample input:

```python
X_test = {
    'id': 786543,
    'radius_mean': 18.00,
    'texture_mean': 11.54,
    'perimeter_mean': 125.4, 
    'area_mean': 998,
    'smoothness_mean': 0.1345,
    'compactness_mean': 0.2345,
    'concavity_mean': 0.2999, 
    'concave points_mean': 0.1500, 
    'symmetry_mean': 0.2500, 
    'fractal_dimension_mean': 0.100,
    'radius_se': 1.234,
    'texture_se': 0.9999, 
    'perimeter_se': 11.333,
    'area_se': 234.3,
    'smoothness_se': 0.001,
    'compactness_se': 0.05, 
    'concavity_se': 0.05, 
    'concave points_se': 0.02,
    'symmetry_se': 0.05,
    'fractal_dimension_se': 0.01,
    'radius_worst': 12.3,
    'texture_worst': 150.1,
    'perimeter_worst': 200.1, 
    'area_worst': 2000,
    'smoothness_worst': 0.1,
    'compactness_worst': 0.2, 
    'concavity_worst': 0.8,
    'concave points_worst': 0.8,
    'symmetry_worst': 0.25,
    'fractal_dimension_worst': 0.1
}
```

## Screen Recording

[Here](https://youtu.be/kmm3lnMMxRA) you can find a quick overview of the Azure ML studio, showcasing the project.
