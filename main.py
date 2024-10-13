import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Example usage
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

