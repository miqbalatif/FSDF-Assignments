import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('c:/Users/Atif/Desktop/Pyt/traintitanic.csv')
#sns.heatmap(train.corr(),annot=True)
#sns.jointplot(x='Pclass',y='Fare',data=train,kind='reg')
#sns.countplot(x='Survived', hue='Pclass',data=train)
#sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
#train['Age'].hist(bins=30,color='red',alpha=0.2)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='winter')
train.drop('Cabin',axis=1,inplace=True)
#plt.show()
pd.get_dummies(train['Embarked'],drop_first=True).head()
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train=pd.concat([train,sex,embark],axis=1)
train.drop('Survived',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)
print(accuracy)
'''from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test)[:,1])'''
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))