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

# Gaussian (RBF) Kernel Function
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

# Polynomial Kernel Function
def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

# Kernelized Perceptron Algorithm
class KernelizedPerceptron:
    def __init__(self, kernel='gaussian', sigma=1.0, degree=3, coef0=1, epochs=10):
        # Kernel type (either 'gaussian' or 'polynomial')
        self.kernel_type = kernel
        self.sigma = sigma      # Bandwidth for the Gaussian kernel
        self.degree = degree    # Degree for the Polynomial kernel
        self.coef0 = coef0      # Coefficient for the Polynomial kernel
        self.epochs = epochs    # Number of iterations
        self.support_vectors = []   # List to store support vectors
        self.alphas = []            # List to store the associated labels

    def kernel(self, x1, x2):
        """Select the kernel function based on the provided kernel type."""
        if self.kernel_type == 'gaussian':
            return gaussian_kernel(x1, x2, sigma=self.sigma)
        elif self.kernel_type == 'polynomial':
            return polynomial_kernel(x1, x2, degree=self.degree, coef0=self.coef0)
        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian' or 'polynomial'.")

    def fit(self, X, y):
        """Train the Kernelized Perceptron."""
        # Training loop over epochs
        for epoch in range(self.epochs):
            print('epoch', epoch)
            for i in range(len(X)):
                prediction = self.predict_single(X[i])
                
                # If the prediction is wrong, add the current sample as a support vector
                if prediction != y[i]:
                    self.support_vectors.append(X[i])
                    self.alphas.append(y[i])

    def predict_single(self, x):
        """Predict a single sample based on the support vectors."""
        result = 0
        for alpha, sv in zip(self.alphas, self.support_vectors):
            result += alpha * self.kernel(x, sv)
        
        return np.sign(result)

    def predict(self, X):
        """Predict for all samples in the dataset."""
        return np.array([self.predict_single(x) for x in X])


# Kernelized Pegasos Algorithm for SVM
class KernelizedPegasos:
    def __init__(self, kernel='gaussian', sigma=1.0, degree=3, coef0=1, lam=0.001, epochs=1000):
        # Kernel type ('gaussian' or 'polynomial')
        self.kernel_type = kernel
        self.sigma = sigma      # Bandwidth for Gaussian kernel
        self.degree = degree    # Degree for Polynomial kernel
        self.coef0 = coef0      # Coefficient for Polynomial kernel
        self.lam = lam          # Regularization parameter
        self.epochs = epochs    # Number of training epochs
        self.alphas = []        # List to store alpha values (for support vectors)
        self.support_vectors = []  # Support vectors
        self.support_vector_labels = []  # Corresponding labels for the support vectors

    def kernel(self, x1, x2):
        """Select the kernel function based on the provided kernel type."""
        if self.kernel_type == 'gaussian':
            return gaussian_kernel(x1, x2, sigma=self.sigma)
        elif self.kernel_type == 'polynomial':
            return polynomial_kernel(x1, x2, degree=self.degree, coef0=self.coef0)
        else:
            raise ValueError("Unsupported kernel type. Choose 'gaussian' or 'polynomial'.")

    def fit(self, X, y):
        """Train the Kernelized Pegasos SVM."""
        n_samples = len(X)
        
        # Initialize alphas to zero (alphas represent weights for each support vector)
        self.alphas = np.zeros(n_samples)

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            print('epoch', epoch)
            print(f"--- {time.time() - start_time:.2f} seconds ---")
            eta = 1 / (self.lam * epoch)  # Learning rate
            
            # Shuffle the training set
            idx = np.random.permutation(n_samples)

            for i in idx:
                x_i = X[i]
                y_i = y[i]
                
                # Compute the kernelized decision function for sample x_i
                decision = sum(self.alphas[j] * y[j] * self.kernel(X[j], x_i) for j in range(n_samples))
                
                # Check the hinge loss condition
                if y_i * decision < 1:
                    self.alphas[i] = (1 - eta * self.lam) * self.alphas[i] + eta
                else:
                    self.alphas[i] = (1 - eta * self.lam) * self.alphas[i]
        
        # After training, store the support vectors (non-zero alphas)
        support_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        self.alphas = self.alphas[support_indices]

    def predict_single(self, x):
        """Predict the label for a single sample."""
        decision = sum(alpha * y_i * self.kernel(sv, x) 
                       for alpha, sv, y_i in zip(self.alphas, self.support_vectors, self.support_vector_labels))
        return np.sign(decision)

    def predict(self, X):
        """Predict labels for a batch of samples."""
        return np.array([self.predict_single(x) for x in X])

# --------------------------------------------------------------------

file_path = 'dataset/data.csv'  # Update with the path to your file)

df = pd.read_csv(file_path)
df.dropna(inplace=True)  # Remove rows with missing values

df.info()
df.head()

sns.pairplot(df, hue='y')  # 'y' is the label column
plt.show()

df.iloc[:, :-1].hist(bins=30, figsize=(10, 10), layout=(3, 4))  # Skips the last column (label)
plt.tight_layout()
plt.show()

# Box plot for each feature (excluding the label)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :-1])
plt.xticks(rotation=90)
plt.title('Box Plots of Features')
plt.show()

X = df.drop(columns='y').values
y = df['y'].values

np.unique(y, return_counts=True)

# Then we split the dataset in training set (60%) and test set (40%) using stratification.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

# --------------------------------------------------------------------

epochs = 1000

print('Train perceptron')
w = perceptron(X_train, y_train, epochs=1000, eta=0.001)
# np.save("perceptron.npy", w)
# Make predictions on the test set
y_pred = predict(X_test, w)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy for Perceptron: {accuracy * 100:.2f}%")

# ---------------------

epochs = np.arange(1, epoch_num+1)
plt.plot(epochs, misclassified_arr)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()

print('Train pegasos')
w, hinge_losses = pegasos(X_train, y_train, epochs)
plot_loss(hinge_losses)
# np.save("pegasos.npy", w)

# Make predictions on the test set
y_pred = predict(X_test, w)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy for Pegasos: {accuracy * 100:.2f}%")

# ---------------------

print('Train pegasos_logistic_loss')
w, log_loss = pegasos_logistic_loss(X_train, y_train, lam=0.001, epochs=2000, eta=0.0001)
plot_loss(log_loss)
# np.save("regularized_log_classification.npy", w)

# Make predictions on the test set
y_pred = predict(X_test, w)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


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
w_perceptron_poly = perceptron(X_poly_train, y_train, epochs=1000, eta=0.01)

# Predict on the expanded test set
y_pred_poly = predict(X_poly_test, w_perceptron_poly)

# Calculate accuracy
accuracy_perceptron_poly = np.mean(y_pred_poly == y_test)
print(f"Perceptron with Polynomial Features (Degree 2) Test Accuracy: {accuracy_perceptron_poly * 100:.2f}%")

# ---------------------

# Train the Pegasos on the expanded features

print('Train the Pegasos on the expanded features')
w_pegasos_poly, hinge_losses_poly = pegasos(X_poly_train, y_train, epochs=700)
print(w_pegasos_poly.shape)

# Predict on the expanded test set
y_pred_poly = predict(X_poly_test, w_pegasos_poly)

# Calculate accuracy
accuracy_pegasos_poly = np.mean(y_pred_poly == y_test)
print(f"Pegasos with Polynomial Features (Degree 2) Test Accuracy: {accuracy_pegasos_poly * 100:.2f}%")

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
compare_weights(w_perceptron_poly, n_features)
compare_weights(w_pegasos_poly, n_features)
compare_weights(w_pegasos_log_loss_poly, n_features)

# --------------------------------------------------------------------

# Kernelized Perceptron

# Create a Kernelized Perceptron instance with the Gaussian kernel
kp_gaussian = KernelizedPerceptron(kernel='gaussian', sigma=1.0, epochs=5)

print('Train kernelized perceptron with Gaussian Kernel')
    
# Train the model with Gaussian kernel
kp_gaussian.fit(X_train, y_train)

# Predict on test data using Gaussian kernel
y_pred_gaussian = kp_gaussian.predict(X_test)

# ---------------------

# Create a Kernelized Perceptron instance with the Polynomial kernel
kp_polynomial = KernelizedPerceptron(kernel='polynomial', degree=3, coef0=1, epochs=5)
    
print('Train kernelized perceptron with polynomial Kernel')

# Train the model with Polynomial kernel
kp_polynomial.fit(X_train, y_train)

# Predict on test data using Polynomial kernel
y_pred_polynomial = kp_polynomial.predict(X_test)

# ---------------------

# Accuracy on test set for Gaussian kernel
acc_gaussian = np.mean(y_test == y_pred_gaussian)
print(f"Gaussian Kernel Accuracy: {acc_gaussian * 100:.2f}%")

# Accuracy on test set for Polynomial kernel
acc_polynomial = np.mean(y_test == y_pred_polynomial)
print(f"Polynomial Kernel Accuracy: {acc_polynomial * 100:.2f}%")

# --------------------------------------------------------------------

# Kernelized Pegasos

# Create Kernelized Pegasos instance with Gaussian kernel
kp_gaussian = KernelizedPegasos(kernel='gaussian', sigma=1.0, lam=0.001, epochs=10)

# Train the model
print('Train kernelized pegasos wiht gaussian model')
kp_gaussian.fit(X_train, y_train)

# Predict with Gaussian kernel
y_pred_gaussian = kp_gaussian.predict(X_test)

# ---------------------

# Create Kernelized Pegasos instance with Polynomial kernel
kp_polynomial = KernelizedPegasos(kernel='polynomial', degree=3, coef0=1, lam=0.01, epochs=10)

# Train the model with Polynomial kernel
print('Train kernelized pegasos wiht polynomial model')
kp_polynomial.fit(X_train, y_train)

# Predict with Polynomial kernel
y_pred_polynomial = kp_polynomial.predict(X_test)

# ---------------------

# Accuracy on test set for Gaussian kernel
acc_gaussian = np.mean(y_test == y_pred_gaussian)
print(f"Gaussian Kernel Accuracy: {acc_gaussian * 100:.2f}%")

# Accuracy on test set for Polynomial kernel
acc_polynomial = np.mean(y_test == y_pred_polynomial)
print(f"Polynomial Kernel Accuracy: {acc_polynomial * 100:.2f}%")
