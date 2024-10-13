# Kernel Linear Classification

## Instructions:

Besides complying with the project’s specifications, it is extremely important that students follow a sound methodology both in the data preprocessing phase and when running the experiments. In particular, no data manipulation should depend on test set information. Moreover, hyperparameter tuning should focus on regions of values where performance trade-offs are explicit. Any implementation must use Python 3 (any other choice must be preliminarily agreed with the teaching assistants).

## Project 1: Kernelized Linear Classification

Download this dataset. The goal is to learn how to classify the  labels based on the numerical features  according to the 1-0 loss, which is the metric you should adopt when evaluating the trained models. Explore the dataset and perform the appropriate preprocessing steps. Please be mindful of data leakage between the training and test sets.

Implement from scratch (without using libraries such as Scikit-learn) the following machine learning algorithms:

1. The Perceptron
2. Support Vector Machines (SVMs) using the Pegasos algorithm
3. Regularized logistic classification (i.e., the Pegasos objective function with logistic loss instead of hinge loss)

Test the performance of these models. Next, attempt to improve the performance of the previous models by using polynomial feature expansion of degree 2. Include and compare the linear weights corresponding to the various numerical features you found after the training phase.

Then, try using kernel methods. Specifically, implement from scratch (again, without using libraries such as Scikit-learn):

1. The kernelized Perceptron with the Gaussian and the polynomial kernels
2. The kernelized Pegasos with the Gaussian and the polynomial kernels for SVM (refer to the kernelized Pegasos paper with its pseudo-code here in Figure 3. Note that there is a typo in the pseudo-code. Identify and correct it.)

Evaluate the performance of these models as well.

Remember that relevant hyperparameter tuning is a crucial part of the project and must be performed using a sound procedure.

Ensure that the code you provide is polished, working, and, importantly, well-documented.

Write a report discussing your findings, with particular attention to the adopted methodology, and provide a thorough discussion of the models’ performance and their theoretical interpretation. Include comments on the presence of overfitting or underfitting and discuss the computational costs.

## Comments during research

## Dataset

Number of 1s == 4992
Number of -1s == 5008

It means that this dataset is balanced and there is no need for undersample or other methods reducing imbalance.
There are no labels other than 1 or -1, so there are no label outliers / missing values.



