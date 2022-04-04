#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

df=pd.read_csv("studentsResponses.csv")
df['Pass/Fail']= df ['Please mention your Previous Semester GPA?'].apply(lambda x: 'Pass' if float(x)>2.3 else 'fail')

from sklearn.preprocessing import LabelEncoder

def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
df = Encoder(df).fillna(df.mean())


"""# **Training Settings**"""

y=df.iloc[:,-1]
x=df.drop(columns=['Please mention your Previous Semester GPA?'],axis=1)

#Train Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 10)

"""# Here we are applying different classifiers to get the one with best accuracy. """
"""# Binary Classification """
"""#
k-Nearest Neighbors
Naive Bayes
Decision Trees
Support Vector Machine
Logistic Regression
"""

# KNN Classifier """

from sklearn.neighbors import KNeighborsClassifier

error_rate = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(15,10))
plt.plot(range(1,50),error_rate, marker='o', markersize=9)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)
# print(metrics.classification_report(y_test,neigh.predict(X_test)))

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
for i in range(1,50):
#   print('Number of clusters are',i,'\n')
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(x, y)
#   print(metrics.classification_report(y_test,neigh.predict(X_test)))
#   print('\n')


"""# Naive_Bayes Classifier"""

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)



#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
# print(metrics.classification_report(y_test,y_pred))


"""# Decision Tree Classifier"""


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# print(metrics.classification_report(y_test,y_pred))


"""# Support Vector Machine Classifier"""

from sklearn.svm import SVC
classifier = SVC()
clf = classifier.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# print(metrics.classification_report(y_test,y_pred))

"""# Linear Model Classifier  -Logistic Regression"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
clf = classifier.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# print(metrics.classification_report(y_test,y_pred))



from sklearn.model_selection import cross_val_score

print ( "KNN",cross_val_score (KNeighborsClassifier(n_neighbors=5),x,y))
print ( "Naive Bayes", cross_val_score(GaussianNB(),x,y))
print ("Decision Tree", cross_val_score(tree.DecisionTreeClassifier(),x,y))
print ("SVC", cross_val_score(SVC(),x,y))
print ("Logistic Regression", cross_val_score(LogisticRegression(),x,y))



# KNN [0.808 0.816 0.8   0.808 0.808]
# Naive Bayes [1.    1.    1.    1.    0.992]
# Decision Tree [1. 1. 1. 1. 1.]
# SVC [0.832 0.832 0.832 0.832 0.824]
# Logistic Regression [0.824 0.832 0.848 0.832 0.84 ]


"""# After Replacing Nan Values with Mean"""

# KNN [0.828125   0.81889764 0.80314961 0.80314961 0.81102362]
# Naive Bayes [1.         1.         1.         1.         0.99212598]
# Decision Tree [1. 1. 1. 1. 1.]
# SVC [0.828125   0.83464567 0.83464567 0.82677165 0.82677165]
# Logistic Regression [0.8203125  0.87401575 0.8503937  0.86614173 0.84251969]

  
# %%
