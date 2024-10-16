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

def perceptron(X_train, y_train, epoch_num):
    # set weights to zero
    w = np.zeros(shape=(1, X_train.shape[1]+1))
    misclassified_arr = []

    for epoch in range(epoch_num):
        misclassified = 0
        for x, label in zip(X_train, y_train):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())

            target = 1.0 if (y > 0) else -1.0
            delta = label - target
            #print("epoch", epoch)
            #print(x, y, label, target, delta)
            if delta: # misclassified
                misclassified += 1
                w += delta * x
        misclassified_arr.append(misclassified)
    return (w, misclassified_arr)

def pegasus(x, y):
    #add bias to sample vectors
    x = np.c_[x,np.ones(len(x))]

    #initialize weight vector
    w = np.zeros(len(x[0]))

    #learning rate 
    eta = 0.001
    #array of indices for shuffling
    order = np.arange(0,len(x),1)
    # Convergence parameters
    margin_current = 0
    margin_previous = -10

    pos_support_vectors = 0
    neg_support_vectors = 0

    not_converged = True
    t = 0 
    start_time = time.time()

    epoch = 0

    while(not_converged):
        margin_previous = margin_current
        t += 1
        pos_support_vectors = 0
        neg_support_vectors = 0
        
        random.shuffle(order)

        # Loop through shuffled samples
        for i in order:  
            prediction = np.dot(x[i],w)
            
            #check for support vectors
            if (round((prediction),1) == 1):
                pos_support_vectors += 1
                #pos support vec found
            if (round((prediction),1) == -1):
                neg_support_vectors += 1
                #neg support vec found
                
            # Misclassification (hinge loss condition)
            # Update weight without regularization
            if (y[i]*prediction) < 1 :
                w = w + eta * y[i] * x[i]
        
        if(t>1000):    
            margin_current = np.linalg.norm(w)
            print("pos SV", pos_support_vectors, "neg SV", neg_support_vectors, "delta margin", margin_current - margin_previous)
            if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
                not_converged = False
        epoch += 1
        if epoch % 10 == 0:
            print('epoch', epoch, 'time', time.time() - start_time)

    #print running time
    print("--- %s seconds ---" % (time.time() - start_time))
    return w

def regularized_log_classification(x, y):
    #add bias to sample vectors
    x = np.c_[x,np.ones(len(x))]

    #initialize weight vector
    w = np.zeros(len(x[0]))

    #learning rate 
    lam = 0.001
    #array of number for shuffling
    order = np.arange(0,len(x),1)
    margin_current = 0
    margin_previous = -10

    pos_support_vectors = 0
    neg_support_vectors = 0

    not_converged = True
    t = 0 
    start_time = time.time()

    epoch = 0

    while(not_converged):
        margin_previous = margin_current
        t += 1
        pos_support_vectors = 0
        neg_support_vectors = 0
        
        eta = 1/(lam*t)
        fac = (1-(eta*lam))*w
        random.shuffle(order)
        for i in order:  
            prediction = np.dot(x[i],w)
            
            #check for support vectors
            if (round((prediction),1) == 1):
                pos_support_vectors += 1
                #pos support vec found
            if (round((prediction),1) == -1):
                neg_support_vectors += 1
                #neg support vec found
                
            #misclassification
            if (y[i]*prediction) < 1 :
                w = fac + eta*y[i]*x[i]            
            #correct classification
            else:
                w = fac
        
        if(t>1000):    
            margin_current = np.linalg.norm(w)
            print("pos SV", pos_support_vectors, "neg SV", neg_support_vectors, "delta margin", margin_current - margin_previous)
            if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
                not_converged = False
        epoch += 1
        if epoch % 10 == 0:
            print('epoch', epoch, 'time', time.time() - start_time)

    #print running time
    print("--- %s seconds ---" % (time.time() - start_time))
    return w


file_path = 'dataset/data.csv'  # Update with the path to your file
features, labels = read_data(file_path)


df = pd.read_csv(file_path)
#print(df.size)
df.dropna(inplace=True)  # Remove rows with missing values
#print(df.size)

df.info()
df.head()

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
print("Labels:", labels[0:10])
ones = []
neg_ones = []
for l in labels:
    if l == 1:
        ones.append(l)
    elif l == -1:
        neg_ones.append(l)
    else:
        print('neither 1 nor -1')

print("Number of +1 labels:", len(ones))
print("Number of -1 labels:", len(neg_ones))


X = df.drop(columns='y').values
y = df['y'].values

np.unique(y, return_counts=True)

# Then we split the dataset in training set (60%) and test set (40%) using stratification.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

epoch_num = 10

w, misclassified_arr = perceptron(X_train, y_train, epoch_num)
np.save("perceptron.npy", w)
# print(w)
# print(misclassified_arr)

epochs = np.arange(1, epoch_num+1)
plt.plot(epochs, misclassified_arr)
plt.xlabel('iterations')
plt.ylabel('misclassified')
#plt.show()

w = pegasus(X_train, y_train)
np.save("pegasus.npy", w)

w = regularized_log_classification(X_train, y_train)
np.save("regularized_log_classification.npy", w)
