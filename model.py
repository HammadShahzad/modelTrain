import pandas as pd

df=pd.read_csv("studentsResponses.csv")

from sklearn.preprocessing import LabelEncoder

#Function for Converting all Category(text) data into numerical

def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
df = Encoder(df).dropna()

#find x='All Data' and y='Previous GPA'
y=df.iloc[:,50]
x=df.drop(columns=['Please mention your Previous Semester GPA?'],axis=1)

#Train Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

"""# Decision Tree Regressor"""

from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf=clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
r_sqaure=metrics.r2_score(y_test,y_pred)
print(r_sqaure)








