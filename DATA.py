from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data from csv
data = pd.read_csv('heart.csv')

#Data preprocessing
#Set non-digital into digital
ChestPainType_map = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}
data['ChestPainType'] = data['ChestPainType'].map(ChestPainType_map)

Sex_map = {'M':0, 'F':1}
data['Sex'] = data['Sex'].map(Sex_map)

RestingECG_map = {'Normal':0, 'ST':1, 'LVH':2}
data['RestingECG'] = data['RestingECG'].map(RestingECG_map)

ExerciseAngina_map = {'N':0, 'Y':1}
data['ExerciseAngina'] = data['ExerciseAngina'].map(ExerciseAngina_map)

ST_Slope_map = {'Up':0, 'Flat':1, 'Down':2}
data['ST_Slope'] = data['ST_Slope'].map(ST_Slope_map)

#drop outlier
data = data.drop(data['RestingBP'][data['RestingBP']==0].index)
def apply_BP(BP):
    if(BP < 90):
        return 0
    elif(BP > 140):
        return 2
    else:
        return 1
data['RestingBP'] = data['RestingBP'].apply(apply_BP)

data = data.drop(data['Cholesterol'][data['Cholesterol']==0].index)
data = data.drop(data['Cholesterol'][data['Cholesterol']>=400].index)

def apply_Chole(chole):
    if(chole >=240):
        return 1
    else:
        return 0
data['Cholesterol'] = data['Cholesterol'].apply(apply_Chole)

def apply_MaxHR(hr):
    if(hr >= 130):
        return 1
    else:
        return 0
data['MaxHR'] = data['MaxHR'].apply(apply_MaxHR)

data = data.drop(data['Oldpeak'][data['Oldpeak']>4].index)
def apply_Oldpeak(Oldpeak):
    if(Oldpeak<=1):
        return 0
    elif(Oldpeak<=2):
        return 1
    elif(Oldpeak<=3):
        return 2
    else: return 3
data['Oldpeak'] = data['Oldpeak'].apply(apply_Oldpeak)

def apply_Age(age):
    if(age <= 40):
        return 0
    elif(age > 65):
        return 2
    else:
        return 1
data['Age'] = data['Age'].apply(apply_Age)
data.reset_index(drop=True, inplace=True)
#Divide into target and training data
target = data['HeartDisease']
del data['HeartDisease']

#Set training and testing dataset
target = np.array(target)
data = np.array(data)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.2)

#DecisionTree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
acc_test = round(clf.score(X_test, Y_test), 4)
acc_train = round(clf.score(X_train, Y_train), 4)
print("DecisionTree:")
print("Test_Acc:", acc_test)
print("Train_Acc:", acc_train)

#RandomForest
clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', criterion='entropy')
clf2.fit(X_train, Y_train)
acc_test = round(clf2.score(X_test, Y_test),4)
acc_train = round(clf2.score(X_train, Y_train),4)
print("RandomForest")
print("Test_Acc:", acc_test)
print("Train_Acc:", acc_train)

feature_name = ['Age', 'Sex', 'ChestPainType', 'RestingBP',
                'Cholesterol', 'FastingBS', 'RestingECG', 
                'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']



#importances
print("DecisionTree:")
print('')
for i in range(len(feature_name)):
    print(feature_name[i]+':'+str(round(clf.feature_importances_[i]*100, 4))+'%')
print('')
print('')
print("RandomForest:")
print('')
for i in range(len(feature_name)):
    print(feature_name[i]+':'+str(round(clf2.feature_importances_[i]*100, 4))+'%')


#10-fold cross-validation
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,data, target,cv=10,scoring='accuracy')
print('')
print('DecisionTree:')
print(scores)
print(round(scores.mean(),4))
print('')
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', criterion='entropy')
scores = cross_val_score(clf,data, target,cv=10,scoring='accuracy')
print('RandomForest:')
print(scores)
print(round(scores.mean(),4))
