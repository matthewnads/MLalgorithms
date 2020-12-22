from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

"""
Written by Matthew Nadarajah July 2020
Github = https://github.com/matthewnads

Logistic Regression Algorithm with ~90% Accuracy on testing sets 
"""
sns.set(style="white")
sns.set(style="whitegrid", color_codes="True")

data = pd.read_csv("Train.csv", header=0)
##visualizing and cleaning up data

print(data['MULTIPLE_OFFENSE'].value_counts())

##sns.countplot(x='MULTIPLE_OFFENSE', data=data, palette='hls')
##print(data.isnull().sum())
##print(data[data['X_12'].isnull()])
data = data.dropna(subset=['X_12'])

##print(data['MULTIPLE_OFFENSE'].value_counts())
##print(data.isnull().sum())
"""
Some more visualizations, comparing variables to our binary... 

sns.countplot(x= 'MULTIPLE_OFFENSE', data = data, palette = 'Set3')

These seem at first glace to show most importance, others not so much. Use DT to confirm 
plt.scatter(data['X_8'], data['MULTIPLE_OFFENSE'])
plt.scatter(data['X_10'], data['MULTIPLE_OFFENSE'])
plt.scatter(data['X_11'], data['MULTIPLE_OFFENSE'])
plt.scatter(data['X_12'], data['MULTIPLE_OFFENSE'])
plt.scatter(data['X_15'], data['MULTIPLE_OFFENSE'])
plt.show()
"""


print(list(data))
x = data.drop(['INCIDENT_ID', 'DATE', 'MULTIPLE_OFFENSE'], axis=1).values
y = data['MULTIPLE_OFFENSE']

print(x.shape)
print(y.shape)
decisionTree = DecisionTreeClassifier(random_state=15, criterion='entropy', max_depth= 10)
decisionTree.fit(x,y)
DTcol = []
DTfi = []
for i,column in enumerate(data.drop(['INCIDENT_ID', 'DATE', 'MULTIPLE_OFFENSE'], axis=1)):
    DTcol.append(column)
    DTfi.append(decisionTree.feature_importances_[i])

FIdataframe  = zip(DTcol, DTfi)
FIdataframe = pd.DataFrame(FIdataframe, columns=['Feature', 'Feature Importance'])
FIdataframe = FIdataframe.sort_values('Feature Importance', ascending= False).reset_index()
print(FIdataframe)
goodFeatures = FIdataframe['Feature'][0:7]

##HO Validation

xHO = data[goodFeatures].values
yHO = data['MULTIPLE_OFFENSE']
print(yHO.shape)
print(xHO.shape)



print(data['MULTIPLE_OFFENSE'].value_counts())
##imbalanced - using under sampling to balance it out

lowerNumber = len(data[data['MULTIPLE_OFFENSE']==0])
higherNumberIndexes = data[data['MULTIPLE_OFFENSE']==1].index
lowerNumberIndexes = data[data['MULTIPLE_OFFENSE']==0].index
randomHigherNumberIndexes = np.random.choice(higherNumberIndexes, lowerNumber, replace = False)

underSampleIndexes = np.concatenate([lowerNumberIndexes, randomHigherNumberIndexes])
underSample = data.loc[underSampleIndexes]
print(underSample.shape)
xHO = underSample[goodFeatures].values
yHO = underSample['MULTIPLE_OFFENSE']
print(xHO.shape)
print(yHO.shape)

sns.countplot(x=yHO, data= underSample)


##train
X_train, X_test, y_train, y_test = train_test_split(xHO, yHO, train_size=0.8, test_size=0.2, random_state=15)

##valid
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=15)
##looking at dist of variables
print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)

sns.countplot(x=y_valid, palette="Set3")
##LG time!
logReg = LogisticRegression(random_state=15, max_iter=3000)
logReg.fit(X_train, y_train)

yPredict = logReg.predict(X_train)
predictProbability = logReg.predict_proba(X_train)

##Scores for training and testing
print("Training Accuracy: ", logReg.score(X_train, y_train))
print("Testing Accuracy: ", logReg.score(X_test, y_test))


##Classification report
print(classification_report(y_train, yPredict))


##Test File

testData = pd.read_csv('Test.csv', header=0)

testData = testData.dropna(subset=['X_12'])

xTest = testData[goodFeatures].values
testPredict = logReg.predict(xTest)
testPredict = np.array(testPredict)
solutionIncidentID = np.array(testData['INCIDENT_ID'])
solutionPrediction = []
print(testPredict)
print(solutionIncidentID)
solutions = zip(solutionIncidentID, testPredict)

solutionsDF = pd.DataFrame(solutions, columns=['INCIDENT_ID', 'MULTIPLE_INCIDENT_PREDICTION'])
print(solutionsDF)
##solutionsDF.to_csv('solutions.csv')