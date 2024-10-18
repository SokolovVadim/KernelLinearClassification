import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# Tools for scaling data, PCA, and standard datasets
from sklearn import preprocessing, decomposition, datasets

# Tools for tracking learning curves and perform cross validation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve, learning_curve

# Function to read the file and process the data
def read_data(file_path):
    features = []
    labels = []

    # Open the file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        # Skip the first line (description/header)
        next(reader)
        # Read each row
        for row in reader:
            # Convert the first 10 values (features) to float, and the last value (label) to int
            x_values = list(map(float, row[:-1]))
            y_value = int(row[-1])
            
            # Append the features and label to respective lists
            features.append(x_values)
            labels.append(y_value)
    
    return features, labels

def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# Predict function for Pegasos
def predict(X, w):
    # Add bias term
    X = np.c_[X, np.ones(len(X))]
    return np.sign(np.dot(X, w))

def perceptron(X, y, epochs=1000, eta=0.01):
    # Add bias term to sample vectors (X -> [X | 1] for bias)
    X = np.c_[X, np.ones(len(X))]

    # Initialize weight vector to zero
    w = np.zeros(len(X[0]))  # w has size equal to number of features (+1 for bias)
    # Array of indices for shuffling
    order = np.arange(0, len(X), 1)

    for epoch in range(epochs):
        # Shuffle data for each epoch to avoid cycles
        random.shuffle(order)

        # Loop through shuffled samples
        for i in order:
            prediction = np.dot(X[i], w)  # Linear prediction
            
            # If y[i] * prediction <= 0, it means misclassification
            if y[i] * prediction <= 0:
                w = w + eta * y[i] * X[i]  # Update the weights if misclassified
        if epoch % 100 == 0:
            print('epoch', epoch)
    return w


def pegasos(x, y, epochs=1000):
    n_samples, n_features = x.shape
    
    # Add bias term to sample vectors
    x = np.c_[x, np.ones(len(x))]  # Adding bias

    # Initialize weight vector
    w = np.zeros(n_features + 1)  # Including bias
    
    # Array of indices for shuffling
    order = np.arange(0, n_samples)

    # List to track hinge loss per epoch
    hinge_losses = []
    
    # Start training
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Shuffle data
        random.shuffle(order)
        
        # Loop through shuffled samples
        for i in order:
            prediction = np.dot(x[i], w)
            
            # Misclassification condition (hinge loss condition)
            if y[i] * prediction < 1:
                # Update weight vector
                w += y[i] * x[i]  # No learning rate or regularization
                
        # Calculate hinge loss for this epoch
        hinge_loss = np.sum([max(0, 1 - y[i] * np.dot(x[i], w)) for i in range(n_samples)])
        hinge_losses.append(hinge_loss)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Hinge Loss: {hinge_loss}')
    
    # Print final running time
    print(f"--- {time.time() - start_time:.2f} seconds ---")
    
    return w, hinge_losses

def pegasos_logistic_loss(x, y, lam=0.001, epochs=1000, eta=0.001):
    # Add bias term to sample vectors
    x = np.c_[x, np.ones(len(x))]

    # Initialize weight vector to zero
    w = np.zeros(len(x[0]))

    # Array of indices for shuffling
    order = np.arange(0, len(x), 1)
    
    logistic_losses = []  # List to track logistic loss per epoch

    for epoch in range(epochs):
        # Shuffle data order
        random.shuffle(order)

        # Loop through shuffled samples
        for i in order:
            prediction = np.dot(x[i], w)  # Linear prediction
            
            # Logistic loss gradient
            z = y[i] * prediction
            gradient = -y[i] * x[i] * (1 / (1 + np.exp(z)))

            # Update weight vector with logistic loss gradient and regularization
            w = (1 - eta * lam) * w - eta * gradient

        # Calculate the logistic loss after each epoch
        logistic_loss = np.sum([np.log(1 + np.exp(-y[i] * np.dot(x[i], w))) for i in range(len(x))])
        logistic_losses.append(logistic_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Logistic Loss: {logistic_loss}')

    # Return the final weight vector and logistic loss history
    return w, logistic_losses


def polynomial_features(X, degree=2):
    # Get number of samples (n) and features (d)
    n_samples, n_features = X.shape
    
    # Initialize list to store expanded features
    expanded_features = []
    
    # Loop through each sample in the dataset
    for x in X:
        # Start with the bias term (constant 1)
        features = [1]
        
        # Add original features (degree 1)
        features.extend(x)
        
        # Add squared terms and cross-products for degree 2
        for i in range(n_features):
            for j in range(i, n_features):
                features.append(x[i] * x[j])
        
        # Append the expanded feature set for this sample
        expanded_features.append(features)
    
    return np.array(expanded_features)


def compare_weights(w, n_features):
    # Assuming we have trained the weights
    # Separate out the bias term from the weights
    weights = w[:-1]  # Excluding the bias term
    bias = w[-1]  # The bias term is the last element
    feature_names = []
    # Plot the weights
    for i in range(n_features):
        feature_names.append('x' + str(i + 1))
    plt.barh(feature_names, weights)
    plt.xlabel('Weight Value')
    plt.ylabel('Feature')
    plt.title('Perceptron Weights for Each Feature')
    plt.show()

    # Print the weights for comparison
    for i, weight in enumerate(weights):
        print(f"Feature {i+1}: Weight = {weight}")
    print(f"Bias term: {bias}")


file_path = 'dataset/data.csv'  # Update with the path to your file
features, labels = read_data(file_path)


df = pd.read_csv(file_path)
#print(df.size)
df.dropna(inplace=True)  # Remove rows with missing values
#print(df.size)

# df.info()
# df.head()

# sns.pairplot(df, hue='y')  # 'y' is the label column
# plt.show()

# df.iloc[:, :-1].hist(bins=30, figsize=(10, 10), layout=(3, 4))  # Skips the last column (label)
# plt.tight_layout()
# plt.show()

# Box plot for each feature (excluding the label)
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df.iloc[:, :-1])
# plt.xticks(rotation=90)
# plt.title('Box Plots of Features')
# plt.show()

# Print the result to verify
#print("Features:", features[0:10])
# print("Labels:", labels[0:10])
# ones = []
# neg_ones = []
# for l in labels:
#     if l == 1:
#         ones.append(l)
#     elif l == -1:
#         neg_ones.append(l)
#     else:
#         print('neither 1 nor -1')

# print("Number of +1 labels:", len(ones))
# print("Number of -1 labels:", len(neg_ones))


X = df.drop(columns='y').values
y = df['y'].values

np.unique(y, return_counts=True)

# Then we split the dataset in training set (60%) and test set (40%) using stratification.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

epochs = 1000

# w = perceptron(X_train, y_train, epochs=1000, eta=0.01)
# np.save("perceptron.npy", w)
# # Make predictions on the test set
# y_pred = predict(X_test, w)

# # Calculate accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"Test Accuracy for Perceptron: {accuracy * 100:.2f}%")

# epochs = np.arange(1, epoch_num+1)
# plt.plot(epochs, misclassified_arr)
# plt.xlabel('iterations')
# plt.ylabel('misclassified')
#plt.show()

#w, hinge_losses = pegasos(X_train, y_train, epochs)
# plot_loss(hinge_losses)
# # np.save("pegasos.npy", w)

# Make predictions on the test set
# y_pred = predict(X_test, w)

# # Calculate accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"Test Accuracy for Pegasos: {accuracy * 100:.2f}%")

# w, log_loss = pegasos_logistic_loss(X_train, y_train, lam=0.001, epochs=2000, eta=0.0001)
# plot_loss(log_loss)
# np.save("regularized_log_classification.npy", w)

# # Make predictions on the test set
# y_pred = predict(X_test, w)

# # # Calculate accuracy
# accuracy = np.mean(y_pred == y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------------------------
# polynomial feature expansion of degree 2

# Transform the original features into polynomial features
X_poly_train = polynomial_features(X_train, 2)
X_poly_test = polynomial_features(X_test, 2)

n_samples, n_features = X_poly_train.shape
print(X_train.shape, y_train.shape)
print(X_poly_train.shape)
print(X_poly_test.shape)

# ---------------------
# Train the Perceptron on the expanded features
# w_perceptron_poly = perceptron(X_poly_train, y_train, epochs=1000, eta=0.01)

# # Predict on the expanded test set
# y_pred_poly = predict(X_poly_test, w_perceptron_poly)

# # Calculate accuracy
# accuracy_perceptron_poly = np.mean(y_pred_poly == y_test)
# print(f"Perceptron with Polynomial Features (Degree 2) Test Accuracy: {accuracy_perceptron_poly * 100:.2f}%")

# ---------------------

# Train the Pegasos on the expanded features

# print('Train the Pegasos on the expanded features')
# w_pegasos_poly, hinge_losses_poly = pegasos(X_poly_train, y_train, epochs=700)
# print(w_pegasos_poly.shape)

# # Predict on the expanded test set
# y_pred_poly = predict(X_poly_test, w_pegasos_poly)

# # Calculate accuracy
# accuracy_pegasos_poly = np.mean(y_pred_poly == y_test)
# print(f"Pegasos with Polynomial Features (Degree 2) Test Accuracy: {accuracy_pegasos_poly * 100:.2f}%")

# ---------------------

# Train the Pegasos with log loss on the expanded features
print('Train the Pegasos with log loss on the expanded features')
w_pegasos_log_loss_poly, logistic_losses = pegasos_logistic_loss(X_poly_train, y_train,lam=0.1, epochs=100, eta=0.000001)

# Predict on the expanded test set
y_pred_poly = predict(X_poly_test, w_pegasos_log_loss_poly)

# Calculate accuracy
accuracy_pegasos_log_loss_poly = np.mean(y_pred_poly == y_test)
print(f"Pegasos with Polynomial Features (Degree 2) Test Accuracy: {accuracy_pegasos_log_loss_poly * 100:.2f}%")


# --------------------------------------------------------------------
# Compare polynomial features
# compare_weights(w_perceptron_poly, n_features)
# compare_weights(w_pegasos_poly, n_features)
compare_weights(w_pegasos_log_loss_poly, n_features)
