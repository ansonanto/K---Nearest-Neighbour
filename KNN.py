import numpy as np  
import pandas as pd  
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("kdata.csv")
X=dataset.iloc[:,0:2].values
y=dataset.iloc[:,2].values

classifier = KNeighborsClassifier(n_neighbors=3)  
classifier.fit(X, y)

X_test=np.array([6,6])
y_pred=classifier.predict([X_test])

print ('General KNN Prediction', y_pred) 
classifier=KNeighborsClassifier(n_neighbors=3,weights='distance') #‘distance’ : weight points by the inverse of their distance. 
#in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
classifier.fit(X,y)

X_test=np.array([6,2])
y_pred=classifier.predict([X_test])
print ('Distance Weighted KNN Prediction', y_pred) 

