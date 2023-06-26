# importing numpy library
import numpy as np
class SVM_classifier():
 # initiating the hyperparameters
 def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

  self.learning_rate = learning_rate
  self.no_of_iterations = no_of_iterations
  self.lambda_parameter = lambda_parameter
 
 # fitting the dataset to SVM Classifier
 def fit(self, X, Y):

  # m  --> number of Data points --> number of rows
  # n  --> number of input features --> number of columns
  self.m, self.n = X.shape

  # initiating the weight value and bias value

  self.w = np.zeros(self.n)

  self.b = 0

  self.X = X

  self.Y = Y

  # implementing Gradient Descent algorithm for Optimization

  for i in range(self.no_of_iterations):
   self.update_weights()


 # function for updating the weight and bias value
 def update_weights(self):
    # label encoding
  y_label = np.where(self.Y <= 0, -1, 1)


  # gradients ( dw, db)
  for index, x_i in enumerate(self.X):

   condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

   if (condition == True):

    dw = 2 * self.lambda_parameter * self.w
    db = 0

   else:

    dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
    db = y_label[index]

   self.w = self.w - self.learning_rate * dw

   self.b = self.b - self.learning_rate * db


 # predict the label for a given input value
 def predict(self, X):

  output = np.dot(X, self.w) - self.b
 
  predicted_labels = np.sign(output)

  y_hat = np.where(predicted_labels <= -1, 0, 1)

  return y_hat 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# loading the data from csv file to pandas dataframe
diabetes_data = pd.read_csv('diabetes.csv')

# print the first 5 rows of the dataframe
diabetes_data.head()

# number of rows and columns in the dataset
diabetes_data.shape


# getting the statistical measures of the dataset
diabetes_data.describe()


diabetes_data['Outcome'].value_counts()

# separating the features and target

features = diabetes_data.drop(columns='Outcome', axis=1)

target = diabetes_data['Outcome']


print(features)

print(target)

scaler = StandardScaler()

scaler.fit(features)

standardized_data = scaler.transform(features)

print(standardized_data)

features = standardized_data
target = diabetes_data['Outcome']

print(features)
print(target)
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state = 2)

print(features.shape, X_train.shape, X_test.shape)

classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)

# training the SVM classifier with training data
classifier.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score on training data = ', training_data_accuracy)

# accuracy on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score on test data = ', test_data_accuracy)

input_data = (5,166,72,19,175,25.8,0.587,51)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
 print('The person is not diabetic')

else:
 print('The Person is diabetic')
