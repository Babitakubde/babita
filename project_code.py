# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:54:22 2019

@author: Ravi
"""


from pandas import Series,DataFrame

# import libs 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# machine learning 
import sklearn
from sklearn import preprocessing
from scipy.stats import pearsonr

# machine learning  - supervised
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

# machine learning  - unsupervised
from sklearn import decomposition
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
# visualization and plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
%matplotlib qt
import warnings
warnings.filterwarnings('ignore')


%pwd
df=pd.read_csv("D:\\statistics\\project\\indian_liver_patient.csv")
df.head()
df.describe()
df.describe(include=['O'])
df.info()
df.shape
# let's look on first entries in the data
df.head(3)
# let's look on target variable - classes imbalanced?
df['Dataset'].value_counts()
# what are the missing values? 
df[df["Albumin_and_Globulin_Ratio"].isnull()]
df["Albumin_and_Globulin_Ratio"].plot(kind='hist')
# fill with median/mean/max/min or ?
df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
# encode gender

le = preprocessing.LabelEncoder()
le.fit(df.Gender.unique())
df['Gender_Encoded'] = le.transform(df.Gender)
df.drop(['Gender'], axis=1, inplace=True)

# correlation plots
g = sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")


# calculate correlation coefficients for two variables
print(pearsonr(df['Total_Bilirubin'], df['Direct_Bilirubin']))

# calculate correlation coefficients for all dataset
correlations = df.corr()

# and visualize
plt.figure(figsize=(10, 10))
g = sns.heatmap(correlations, cbar = True, square = True, annot=True, fmt= '.2f', annot_kws={'size': 10})


# based on correlation, you can exclude some highly correlated features


# pair grid allows to visualize multi-dimensional datasets
 g = sns.PairGrid(df, hue="Dataset", vars=['Age','Gender_Encoded','Total_Bilirubin','Total_Protiens'])
g.map(plt.scatter)
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(label='count',x='Dataset',data=df)
sns.catplot(data=df,y='Age',x='Gender_Encoded',hue='Dataset',jitter=0.4)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")
sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")
df.corr()
#Results of Analysis:
#Age and Gender affect the occurence of disease.
#Some features are directly correlated like Total_Bilirubin and Direct_Bilirubin, Aspartate_Aminotransferase and Alamine_Aminotransferase, Total_Protiens and Albumin.
M#ale has more the no of liver disease than female.
#Since gender is categorical we need to convert it to numeric data.
df['Dataset']=df['Dataset'].map({1:0,2:1}).astype(int)
df

X = df.drop(['Gender_Encoded','Dataset','Direct_Bilirubin','Aspartate_Aminotransferase'], axis=1)
y = df['Dataset']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=101)

#Logistic Regression
from sklearn.linear_model import LogisticRegression   
logmodel = LogisticRegression()
logmodel.fit(xTrain,yTrain)
#Finding the performance of model

lg_pred = logmodel.predict(xTest)
#confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(yTest,lg_pred)
cmlog= confusion_matrix(yTest, lg_pred)
cmlog[1,1]
cmlog.sum()
acculog= (cmlog[0,0]+cmlog[1,1])/cmlog.sum()*100
round(acculog)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

lg_roc_auc = roc_auc_score(yTest,logmodel.predict(xTest))
round(lg_roc_auc*100)

#Probability
Lgprob = logmodel.predict_proba(xTest)

lgfpr,lgtpr,thresholds = roc_curve(yTest,logmodel.predict_proba(xTest)[:,1])

#Plot ROC and AUC

plt.figure()
plt.plot(lgfpr,lgtpr,label='Logistic Regression (area = %0.2f)' %lg_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right") #location of Legend by default is left top
plt.show()



#support Vector Machines (SVM) ensemble model faster

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

svc= OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear',
                                               probability=True)))
svc.fit(xTrainS,yTrain)

Y_pred_SVM = svc.predict(xTestS)
confusion_matrix(yTest,Y_pred_SVM)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

svm_roc_auc= roc_auc_score(yTest,svc.predict(xTest))
fpr,tpr,thresholds = roc_curve(yTest,svc.predict_proba(xTest)[:,1])

plt.figure()
plt.plot(fpr,tpr,label='SVM (area = %0.2f)' %svm_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right") #location of Legend by default is left top
plt.show()


#Random forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators= 100, oob_score = True, random_state=123) #n_estimators = 100 estimates , oob_score = out of bag score 

random_forest.fit(xTrain,yTrai
                  
                  n)
Y_prediction = random_forest.predict(xTest)
random_forest.score(xTrain,yTrain)
#Random Forest confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(yTest, Y_prediction)

rf_roc_auc = roc_auc_score(yTest,random_forest.predict(xTest))
round(rf_roc_auc*100)

#Random forest Probability
rfprob = random_forest.predict_proba(xTest)

rffpr,rftpr,thresholds = roc_curve(yTest,random_forest.predict_proba(xTest)[:,1])
plt.figure()
plt.plot(rffpr,rftpr,label='Radom Forest (area = %0.2f)' %rf_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right") #location of Legend by default is left top
plt.show()





#decision tree
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(xTrain,yTrain)
Y_Pred_DT = decision_tree.predict(xTest)
confusion_matrix(yTest, Y_Pred_DT)


dtrob = decision_tree.predict_proba(xTest)
dt_roc_auc = roc_auc_score(yTest,decision_tree.predict(xTest))
dtfpr,dttpr,thresholds = roc_curve(yTest,decision_tree.predict_proba(xTest)[:,1])
plt.figure()
plt.plot(dtfpr,dttpr,label='Decision Tree (area = %0.2f)' %dt_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right") #location of Legend by default is left top
plt.show()
### ROC & AUC Cuve

plt.figure()
plt.plot(fpr,tpr,label='SVM(area= %0.2f)'%svm_roc_auc)
plt.plot(lgfpr,lgtpr,label='LG(area= %0.2f)'%lg_roc_auc)
plt.plot(rffpr,rftpr,label='RF(area= %0.2f)'%rf_roc_auc)
plt.plot(dtfpr,dttpr,label='DT(area= %0.2f)'%dt_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis

plt.legend(loc='lower right',fontsize='xx-large')


'''KNN'''

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=39)
knn.fit(xTrain,yTrain)

Y_pred_knn = knn.predict(xTest)
confusion_matrix(yTest,Y_pred_knn)
#tuning KNN
error = []
#calculating error for k values between 1 and 40
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(xTrain,yTrain)
    pred_i = knn.predict(xTest)
    error.append(np.mean(pred_i != yTest))
    
plt.figure()
plt.plot(error,marker='o')
plt.xticks(np.arange(0,40,step=1))
plt.title('Error Rate K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
'''n_neighbors = 22 because it is the lowest error
or 39 because it is giving the highest AUC'''

#Plot ROC and AUC
knn_roc_auc = roc_auc_score(yTest,knn.predict(xTest))
round(knn_roc_auc*100)
knnprob = knn.predict_proba(xTest)
knnfpr,knntpr,thresholds = roc_curve(yTest,knn.predict_proba(xTest)[:,1])

plt.figure()
plt.plot(knnfpr,knntpr,label='KNN (area = %0.2f)' %knn_roc_auc)
plt.plot([0,1],[0,1],'r--') #r means color red and -- means style of line
plt.xlim([0.0,1.0]) #limit of x axis
plt.ylim([0.0,1.0]) #limit of y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.legend(loc="lower right") #location of Legend by default is left top
plt.show()

