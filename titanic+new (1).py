import pandas as pd


#importing data into pandas
train_data=pd.read_csv("C:\\Users\\Vishal\\Desktop\\titanic\\train.csv")# user path
test_data=pd.read_csv("C:\\Users\\Vishal\\Desktop\\titanic\\test.csv")#user path 
full_data=[train_data,test_data]
train_data.describe()


#info about the train data
train_data.info()


#working with data
#Pclass
train_data[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()

#sex
train_data[['Sex','Survived']].groupby('Sex',as_index=False).mean()



train_data['Family'] =train_data["Parch"] + train_data["SibSp"]
train_data[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()



#Embarked details
train_data['Embarked']=train_data['Embarked'].fillna('S')
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


#age details
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())
train_data['Age']=train_data['Age'].astype(int)    
train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)
train_data[['CategoricalAge', 'Survived']].groupby('CategoricalAge', as_index=False).mean()




#fare details
train_data['Fare']=train_data['Fare'].fillna(train_data['Fare'].median())  
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4)
train_data[['CategoricalFare', 'Survived']].groupby('CategoricalFare', as_index=False).mean()




#dropping the name variable
train_data=train_data.drop(['Name'],axis=1)
#removing the cabin
train_data=train_data.drop(['Cabin'],axis=1)
train_data



#removing ticket info
train_data=train_data.drop(['Ticket'],axis=1)
train_data


#cleaning the data
train_data['Sex']=train_data['Sex'].map({'female': 0, 'male': 1}).astype(int)
train_data['Embarked']=train_data['Embarked'].map({'S': 0, 'C': 1, 'Q':2}).astype(int)
train_data['CategoricalAge']=train_data['CategoricalAge'].map({'(-0.08, 16]': 0, '(16, 32]': 1, '(32, 48]':2,'(48, 64]':3,'(64, 80]':4}).astype(int)
train_data['CategoricalFare']=train_data['CategoricalFare'].map({'[0, 7.91]': 0, '(7.91, 14.454]': 1, '(14.454, 31]':2,'(31, 512.329]':3}).astype(int)
train_data



#dropping other age fare parch sibsp
train_data=train_data.drop(['Age','Fare','Parch','SibSp'],axis=1)
train_data


from sklearn.model_selection import train_test_split
predictors = train_data.drop(['Survived', 'PassengerId'], axis=1)
target = train_data["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)




# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)



# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)




# Support Vector Machines
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[18]:

# Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)




#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)




# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)




# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)



# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)




#results in data frame
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)





