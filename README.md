# Kernel Linear Classification

## Instructions:

Besides complying with the project’s specifications, it is extremely important that students follow a sound methodology both in the data preprocessing phase and when running the experiments. In particular, no data manipulation should depend on test set information. Moreover, hyperparameter tuning should focus on regions of values where performance trade-offs are explicit. Any implementation must use Python 3 (any other choice must be preliminarily agreed with the teaching assistants).

## Project 1: Kernelized Linear Classification

Download this dataset. The goal is to learn how to classify the  labels based on the numerical features  according to the 0-1 loss, which is the metric you should adopt when evaluating the trained models. Explore the dataset and perform the appropriate preprocessing steps. Please be mindful of data leakage between the training and test sets.

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

## SVM Pegasos without regularization

Functionality:

    The algorithm iteratively updates the weight vector w based on misclassifications.
    There’s no regularization (penalizing large weights), so the focus is purely on minimizing hinge loss.
    The loop continues until it meets a convergence criterion, checking for the stabilization of the margin and the presence of support vectors.

Performance:

    Without regularization, the model might overfit the data, especially if the dataset contains noise or outliers.
    This algorithm is faster since no regularization factor is computed, but it may result in a model that performs worse on unseen data due to overfitting.


## Performance

### Perceptron 1000 epochs

Test Accuracy for Perceptron: 69.47%

### Pegasos hinge loss 1000 epochs

epoch 1000 time 78.50806331634521
pos SV 20 neg SV 25 delta margin 5.933144820283454
pos SV 16 neg SV 18 delta margin -0.0018304066875485248
--- 78.66581273078918 seconds ---
Test Accuracy for Pegasos: 71.10%

### Pegasos log loss regularized 1000 epochs

Epoch 800, Logistic Loss: 7243.350702299096
Epoch 900, Logistic Loss: 8456.289560211537
Test Accuracy: 68.88%


## Polynomial feature expansion of degree 2

### Perceptron

To improve the performance of the Perceptron by using polynomial feature expansion of degree 2, we can transform the input features into a higher-dimensional space by adding polynomial combinations of the features. This transformation allows the Perceptron to learn more complex decision boundaries, potentially improving performance, especially for non-linearly separable data.

Steps to Apply Polynomial Expansion to Perceptron:

1. Expand the Features: Use polynomial feature transformation to expand the feature space.
2. Train the Perceptron: Train the standard Perceptron algorithm on the expanded feature set.
3. Evaluate the Model: After training, evaluate the model on a test set.

Perceptron with Polynomial Features (Degree 2) Test Accuracy: 63.32%
Perceptron with Polynomial Features (Degree 2) Test Accuracy: 78.88%

### Pegasos


## Interpret and Compare Weights

The magnitude and sign of each weight provide insights into how each feature influences the prediction. Larger weights (positive or negative) suggest that the feature has a stronger influence.

Interpret the Weights:

1. The weights indicate how much each feature influences the prediction.
- Positive weight: The feature increases the likelihood of a positive class.
- Negative weight: The feature decreases the likelihood of a positive class.
- Magnitude: A higher magnitude means a stronger influence (either positive or negative).
2. For polynomial features, weights corresponding to squared or interaction terms give insight into how combinations of features influence the model.
3. Compare the Weights:
we can create a comparison table or plot the weights to visually analyze their relative importance.

How to Compare:

1. Positive vs Negative Weights: Features with positive weights push predictions towards one class, while negative weights push towards the other class.
2. Magnitude of Weights: Features with larger absolute values for weights are more influential in the model's predictions.
3. Bias Term: The bias term adjusts the overall prediction threshold.

Working with Polynomial Features:

When we've applied polynomial feature expansion, the weights will also correspond to the new polynomial features (like squared terms or interaction terms). We can compare the relative importance of the original features and their polynomial transformations by analyzing the weights in the expanded feature set.

Perceptron:

Perceptron with Polynomial Features (Degree 2) Test Accuracy: 79.25%
Feature 1: Weight = -0.5100000000000002
Feature 2: Weight = 13.893606617998444
Feature 3: Weight = 12.34944521603801
Feature 4: Weight = -23.620586318666938
Feature 5: Weight = -16.723665304206868
Feature 6: Weight = 20.439116407026603
Feature 7: Weight = -19.55637538601126
Feature 8: Weight = 5.859894250597445
Feature 9: Weight = 22.073887553861294
Feature 10: Weight = -0.4207287224345229
Feature 11: Weight = 18.288911048778687
Feature 12: Weight = -440.1228692903385
Feature 13: Weight = 10449.00023563632
Feature 14: Weight = 838.1582584491596
Feature 15: Weight = -992.5962240148892
Feature 16: Weight = -1092.9287181757934
Feature 17: Weight = 585.4232864810298
Feature 18: Weight = 2921.6880503414905
Feature 19: Weight = 10405.437061547382
Feature 20: Weight = 4180.693645010309
Feature 21: Weight = -199.08571063099467
Feature 22: Weight = 20529.57744854524
Feature 23: Weight = -226.0690591124983
Feature 24: Weight = 905.7776466889638
Feature 25: Weight = -2214.9103668510725
Feature 26: Weight = 1342.6795502602092
Feature 27: Weight = 7289.6768963393615
Feature 28: Weight = 8574.329348154371
Feature 29: Weight = 41420.20029880763
Feature 30: Weight = 545.1776367351339
Feature 31: Weight = 292.9082795815996
Feature 32: Weight = -1111.92127989384
Feature 33: Weight = 2257.343594228754
Feature 34: Weight = -1886.9173870681873
Feature 35: Weight = 476.81386364732
Feature 36: Weight = 1322.7772104272426
Feature 37: Weight = -41.65687365168206
Feature 38: Weight = 367.1786962335269
Feature 39: Weight = -842.7611561309117
Feature 40: Weight = 1003.308902136254
Feature 41: Weight = -583.1488940527182
Feature 42: Weight = -39.11542028126787
Feature 43: Weight = -6825.020692757325
Feature 44: Weight = -1652.0067190420716
Feature 45: Weight = 313.30223296471854
Feature 46: Weight = 308.52600541368537
Feature 47: Weight = -208.09168018106948
Feature 48: Weight = 148.19648316854085
Feature 49: Weight = 290.46668529973124
Feature 50: Weight = -798.4759240548976
Feature 51: Weight = -1333.899934730419
Feature 52: Weight = -54.92653869477473
Feature 53: Weight = 130.84757432021223
Feature 54: Weight = 846.9319907559511
Feature 55: Weight = -0.7710416371249302
Feature 56: Weight = 1036.256310407171
Feature 57: Weight = 1217.2809243570548
Feature 58: Weight = 1791.0710245011815
Feature 59: Weight = 7559.30562758123
Feature 60: Weight = -229.0708776452436
Feature 61: Weight = 1931.798055182003
Feature 62: Weight = 10511.451647626614
Feature 63: Weight = -468.3370606962206
Feature 64: Weight = 15740.331686079888
Feature 65: Weight = -43.1892024137251
Feature 66: Weight = -500.2955449853478
Bias term: -0.5100000000000002

Pegasos parametrized

Pegasos with Polynomial Features (Degree 2) Test Accuracy: 70.85%
Feature 1: Weight = -3.970061386426988e-05
Feature 2: Weight = 0.001095317271366397
Feature 3: Weight = -7.940566218474924e-05
Feature 4: Weight = -0.0020457662367589593
Feature 5: Weight = -0.0015489438315632832
Feature 6: Weight = 0.002160636661891941
Feature 7: Weight = -0.0016451006678886086
Feature 8: Weight = 0.0006580976481293098
Feature 9: Weight = 0.0015935009234747772
Feature 10: Weight = 0.00012053334426899961
Feature 11: Weight = 0.0009957174575671004
Feature 12: Weight = 0.007391329778476694
Feature 13: Weight = 0.13932715117964795
Feature 14: Weight = 0.10563266393214325
Feature 15: Weight = -0.01176088082471361
Feature 16: Weight = -0.007988941118168743
Feature 17: Weight = 0.004508085600470771
Feature 18: Weight = 0.05101798829303752
Feature 19: Weight = 0.12358775923783985
Feature 20: Weight = 0.06789912760083457
Feature 21: Weight = -0.05580294073794776
Feature 22: Weight = 0.3310973087757115
Feature 23: Weight = -0.026203383485741232
Feature 24: Weight = 0.011525841532660765
Feature 25: Weight = -0.02715659295713639
Feature 26: Weight = 0.016952768654389473
Feature 27: Weight = 0.12772260108153627
Feature 28: Weight = 0.13968633783908843
Feature 29: Weight = 0.5081007406377509
Feature 30: Weight = 0.020146942293744457
Feature 31: Weight = -0.013555186130282604
Feature 32: Weight = -0.15210430201125302
Feature 33: Weight = 0.2181755029038255
Feature 34: Weight = -0.16345935599484712
Feature 35: Weight = 0.06438294664665176
Feature 36: Weight = 0.15090864912406443
Feature 37: Weight = 0.010168256853445516
Feature 38: Weight = -0.006339117106483183
Feature 39: Weight = -0.002021356737361367
Feature 40: Weight = 0.007707890230111892
Feature 41: Weight = -0.003186411677208136
Feature 42: Weight = -0.00030769615608042917
Feature 43: Weight = -0.06802049058459103
Feature 44: Weight = -0.017078393463345973
Feature 45: Weight = 0.08191092945068301
Feature 46: Weight = 0.003529186199737533
Feature 47: Weight = -0.0022671434325822323
Feature 48: Weight = 0.0024214474790437515
Feature 49: Weight = 0.005492032226205501
Feature 50: Weight = -0.01048155194773688
Feature 51: Weight = -0.1220032616941202
Feature 52: Weight = -0.0007386784823134753
Feature 53: Weight = 0.0014839345948941011
Feature 54: Weight = 0.007861693358466615
Feature 55: Weight = 0.001820523860847676
Feature 56: Weight = 0.09065341868277557
Feature 57: Weight = 0.04709080614693652
Feature 58: Weight = 0.050312817153126785
Feature 59: Weight = 0.12352486381687477
Feature 60: Weight = -0.03524564182614587
Feature 61: Weight = 0.05931608477982823
Feature 62: Weight = 0.15009899908352062
Feature 63: Weight = -0.08151798429771688
Feature 64: Weight = 0.22705971980055115
Feature 65: Weight = -0.005648597701579488
Feature 66: Weight = 0.011242383528421064
Bias term: -3.970061386426988e-05


## Kernel

### Kernelized Gaussian perceptron

Kernelized Perceptron Algorithm

Tuning:

sigma (bandwidth): Controls how much influence a support vector has over the prediction. A smaller σσ results in more localized influence.
epochs: Increasing the number of epochs can help the Perceptron converge if the data is not linearly separable in the original space.

This implementation works for binary classification and uses the Gaussian kernel to implicitly map the data to a higher-dimensional space where a linear decision boundary can be found.


Train kernelized perceptron with Gaussian Kernel
epoch 0
epoch 1
epoch 2
epoch 3
epoch 4
Train kernelized perceptron with polynomial Kernel
epoch 0
epoch 1
epoch 2
epoch 3
epoch 4
Gaussian Kernel Accuracy: 88.75%
Polynomial Kernel Accuracy: 71.40%

## kernelized Pegasos

The kernelized Pegasos with the Gaussian and the polynomial kernels for SVM.


Key Steps in the Code:

    Kernel Functions: gaussian_kernel and polynomial_kernel compute similarity between points based on the chosen kernel.

    Kernelized Pegasos SVM:
        fit() trains the model by updating alphas, which are weights for the support vectors.
        During training, the kernel function replaces the dot product.

    Predict Function:
        For prediction, the decision function is computed using the support vectors and their associated alphas and labels.

    Support Vectors:
        After training, only the support vectors (non-zero alphas) are retained.

Hyperparameters to Tune:

    sigma (for Gaussian kernel): Determines the width of the Gaussian kernel.
    degree (for Polynomial kernel): Degree of the polynomial.
    lam: Regularization parameter.
    epochs: Number of iterations over the data.

Train kernelized pegasos wiht gaussian model
epoch 1
--- 0.00 seconds ---
epoch 2
--- 172.38 seconds ---
epoch 3
--- 349.58 seconds ---
epoch 4
--- 523.27 seconds ---
epoch 5
--- 695.48 seconds ---
epoch 6
--- 879.39 seconds ---
epoch 7
--- 1061.38 seconds ---
epoch 8
--- 1234.70 seconds ---
epoch 9
--- 1409.54 seconds ---
epoch 10
--- 1585.58 seconds ---
Train kernelized pegasos wiht polynomial model
epoch 1
--- 0.00 seconds ---
epoch 2
--- 70.25 seconds ---
epoch 3
--- 140.44 seconds ---
epoch 4
--- 210.35 seconds ---
epoch 5
--- 280.62 seconds ---
epoch 6
--- 350.73 seconds ---
epoch 7
--- 420.80 seconds ---
epoch 8
--- 491.37 seconds ---
epoch 9
--- 563.94 seconds ---
epoch 10
--- 634.11 seconds ---
Gaussian Kernel Accuracy: 89.03%
Polynomial Kernel Accuracy: 57.95%
