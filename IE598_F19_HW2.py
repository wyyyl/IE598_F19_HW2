#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import the treasury data and show the class labels of target y
df = pd.read_csv('C:/Users/Taki/Desktop/Treasury Squeeze test - DS1.csv',header=None)
df =df.values
X=np.array(df[1:,0:8])
y=np.array(df[1:,9])
print('Class labels:',np.unique(y))


# In[2]:


#split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=33,stratify=y)


# In[3]:


from sklearn.neighbors import KNeighborsClassifier
#following codes are from the datacamp online courses
#KNN Classifier
#setup arrays to store train and test accuracies
neighbors=np.arange(1,26)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))

#loop over different values of k
for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy[i]=knn.score(X_train,y_train)
    test_accuracy[i]=knn.score(X_test,y_test)


# In[5]:


#genearte a plot
import matplotlib.pyplot as plt
plt.title('k-NN:Varying Number of Neighbors')
plt.plot(neighbors,test_accuracy,label='Testing Accuracy')
plt.plot(neighbors,train_accuracy,label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1,26,2))
plt.show
#according to the plot, we can claim that k=16 is the best choice
#Since when k=16, the classifier has a highest Testing Accuracy Score and similar Training Accuracy Score


# In[6]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the decision surface
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1 
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=colors[idx],marker=markers[idx], label=cl, edgecolor='black')

# highlight test samples
    if test_idx:
# plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],c='', edgecolor='black', alpha=1.0,linewidth=1, marker='o',s=100, label='test set')


# In[7]:


#Classify the first two columns of data with a DecisionTree Classifier
from sklearn.preprocessing import StandardScaler
df1 = pd.read_csv('C:/Users/Taki/Desktop/Treasury Squeeze test - DS1.csv',header=None)
X1=df1.iloc[1:,[0,1]]
y1=df1.iloc[1:,9]
y1 = pd.factorize(y1)[0].tolist()
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=1,stratify=y1)
sc = StandardScaler()
sc.fit(X1_train)
X_train_std = sc.transform(X1_train)
X_test_std = sc.transform(X1_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y1_train, y1_test))
kn = KNeighborsClassifier(n_neighbors=16)
kn.fit(X_train_std, y1_train)
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=kn,test_idx=range(105,150))
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')
plt.legend(loc='upper left')
plt.show()


# In[8]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#Fit DecisionTree Classifier to the training set
dt = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=3)
dt.fit(X_train,y_train)
#predict test set labels and show its accuracy rate
y_pred=dt.predict(X_test)
print(y_pred[:])
print("Test set accuracy:{:.2f}".format(accuracy_score(y_test,y_pred)))


# In[9]:


#Classify the first two columns of data with a DecisionTree Classifier
dt1=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
dt1.fit(X_train_std, y1_train)
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=dt1,test_idx=range(105,150))
plt.xlabel('price_crossing')
plt.ylabel('price_distortion')
plt.legend(loc='upper left')
plt.show()

print("My name is Yulong Wang")
print("My NetID is yulongw2")
print("I hereby certify that I have read the University policy on Academic Intergrity and that I a not in violation")
