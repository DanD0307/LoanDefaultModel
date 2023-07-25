import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# Load dataset from the csv file
col_names = ['Employed', 'Bankbalance', 'AnnualSalary', 'Defaulted']
data = pd.read_csv("defaults.csv", header=None, names=col_names)

# Split dataset into features and target variable
feature_cols = ['Employed', 'Bankbalance', 'AnnualSalary']
X = data[feature_cols]  # Features
y = data['Defaulted']    # Target variable

# The next step is to Apply SMOTE to upsample the minority class (which is the non-defaulted customers)
# SMOTE (Synthetic Minority Over-sampling Technique) is a technique used to upsample the minority class in an 
# imbalanced dataset by generating synthetic samples. 
# It aims to balance the class distribution by creating synthetic instances for the minority class,
# which helps improve the performance of machine learning models on imbalanced data.
#
# The basic idea behind SMOTE is to create synthetic samples by using interpolation between the feature vectors of 
# existing minority class instances.
#
# Luckily for me the SMOTE library from imblearn does all the heavy lifting of generating this more balanced data.
    
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the upsampled data into training and testing sets. 75 percent is used for training the rest is for testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=16)

# Instantiate the logistic regression model
logreg = LogisticRegression(random_state=18, max_iter=1000)

# Fit the model with data
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Model Analysis
# Here I generate a Confusion Matrix to help evaluate the usefulness of my logistic regression model
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cnf_matrix)

# Create a heatmap for confusion matrix visualization
class_names = ['Good', 'Defaulted']
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g') #YLGnBu stands for Yellow-Green-Blue
plt.xticks(np.arange(len(class_names)), class_names)
plt.yticks(np.arange(len(class_names)), class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# Here I calculate the Gini Coefficient which was a team-wide practice in the 
# Customer Modelling team at Lloyds Banking Group for our models

# The Gini is measure of the inequality of a distribution, and in the context of classification,
# models (like my logistic regression), it quantifies the quality of the model's predictions.


# The inputs for the gini calculator are the true labels of the test set (y_true) and the predicted
# probabilities for defaulted from the logistic regression model
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
def gini_coefficient_calculator(y_true, y_pred_prob):
    # Sort the probabilities and corresponding true labels in descending order
    # This is required to properly calculate a cumulative sum
    sorted_indices = np.argsort(y_pred_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_prob_sorted = y_pred_prob[sorted_indices]

    # Cumulative sums of the true labels
    cumsum_true = np.cumsum(y_true_sorted)
    cumsum_total = cumsum_true[-1]
    cumsum_index = np.arange(1, len(y_true_sorted) + 1)

    # Calculate A and B for the Gini coefficient
    # We need A and B to measure the area between the Lorenz curve which represents the cumulative
    # distribution of positive instances) and the diagonal line (perfect equality).
    A = np.sum(cumsum_true * y_pred_prob_sorted) / cumsum_total - (cumsum_index / len(y_true_sorted)) * y_pred_prob_sorted
    B = np.sum(cumsum_true) / cumsum_total - (cumsum_index / len(y_true_sorted))

    # Calculate the Gini coefficient. Value between 0 and 1
    gini = np.sum(A) / np.sum(B)
    return gini

# Calculate the Gini coefficient of the logistic regression model

gini_coefficient_value = gini_coefficient_calculator(y_test.values, y_pred_prob)
print("Gini coefficient:", gini_coefficient_value)


# Initialize the explainer with the logistic regression model and the training data
# The explainer is a pre-computation step that prepares the model to be used 
# for Shapley value calculations.
explainer = shap.Explainer(logreg, X_train)

# Calculate Shapley values for the test set using the explainer. 
shap_values = explainer(X_test)

# Summary plot of Shapley values to help visualize the data
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, class_names=class_names)

