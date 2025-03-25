# Evaluation Metrics Calculation for Classification Models

## Project Description

This project aims to calculate the primary metrics used to evaluate data classification models, including **accuracy**, **sensitivity (recall)**, **specificity**, **precision**, and **F-score**. The implementation of these metrics will be carried out using their respective formulas and specific methods.

The basis for calculating these metrics will be a **confusion matrix**, which can be chosen arbitrarily to facilitate understanding the functioning of each metric. The project allows for flexibility in selecting the confusion matrix, allowing you to explore how different matrices influence the computed metrics.

## Metrics to be Calculated

- **Accuracy**: The proportion of correct predictions (both true positives and true negatives) over all predictions.
  
- **Sensitivity (Recall)**: The rate of true positives correctly identified by the model.
  
- **Specificity**: The rate of true negatives correctly identified by the model.
  
- **Precision**: The proportion of true positive predictions out of all positive predictions made by the model.
  
- **F-score**: The harmonic mean of precision and sensitivity, balancing the two metrics.

## Confusion Matrix

A confusion matrix is used to calculate these metrics. The matrix contains the following values:

- **VP (True Positives)**: The cases where the model correctly predicted a positive class.
- **FN (False Negatives)**: The cases where the model incorrectly predicted a negative class when it was actually positive.
- **FP (False Positives)**: The cases where the model incorrectly predicted a positive class when it was actually negative.
- **VN (True Negatives)**: The cases where the model correctly predicted a negative class.

## Objective

The goal is to understand how each metric functions by implementing and calculating the values based on the chosen confusion matrix.

## Resources

- **Table 1**: Overview of the metrics used to evaluate classification methods (with explanations of terms like VP, FN, FP, VN, and their corresponding formulas).
