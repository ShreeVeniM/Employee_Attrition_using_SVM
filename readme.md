PROJECT OVERVIEW
Introduction
This project focuses on predicting employee attrition using Support Vector Machines (SVM) and Logistic Regression. The analysis includes data loading, preprocessing, model training, evaluation, and visualization.

Directory Structure
.git: Contains Git version control data.
.gitattributes: Git attributes file.
charts: Directory for storing generated charts and visualizations.
main.py: The main script to run the project.
requirements.txt: Lists the Python dependencies.
src: Source code directory containing modules for various tasks.
Key Components
main.py
The central script that coordinates the following tasks:

Data Loading:

Loads employee attrition data from src/dataset/HR_Employee_Attrition.xlsx.
Data Preprocessing:

Cleans and preprocesses the loaded data.
Splits the data into training and testing sets.
Visualization:

Plots histograms for numerical features.
Plots categorical distributions for categorical features.
Plots a correlation heatmap.
Model Training and Evaluation:

Trains and evaluates Logistic Regression and SVM models with different kernels (linear, RBF, polynomial).
Evaluates the models using appropriate metrics.
Generates evaluation metrics for both training and testing sets.
requirements.txt
Specifies the Python libraries required for the project:

pandas
numpy
matplotlib
seaborn
scikit-learn

Conclusion
This project provides a comprehensive analysis and prediction of employee attrition using SVM and Logistic Regression. It includes detailed visualizations to aid in understanding the dataset and the model's performance. ​