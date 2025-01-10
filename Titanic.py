#Titanic Case study solved by logistic regression 
import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    #step:1 Load data 
    titanic_data=pd.read_csv('MarvellousTitanicDataset.csv')

    print("First 5 entries of dataset :\n",titanic_data.head())
    print("Columns Of Dataset :",titanic_data.columns)
    print("Dimension of dataset :",titanic_data.shape)
    print("Number of passengers :",str(len(titanic_data)))

    #Step 2 :- Analyze data
    print("Visualization : Survived and non survived passengers")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target).set_title("Survived and non survived passengers") 
    show()
    
    
    print("Visualisation : Survived and non survived passengers based on gender")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and Non survived based on gender")
    show()

    print("Visualization : survived and non survived passenger based on the passenger class")
    figure()    
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passengers based on passenger class")
    show()

    print("Visualization :Survived or non survived based on Age")
    figure()
    # target="Survived"
    # countplot(data=titanic_data,x=target,hue="Age").set_title("Survived and Non survived based on age")
    # show()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived based on age ")
    show()

    print("Visualization : Survived and non survived based on fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non survived based on Fare")
    show()

    #Step 3: Data Cleaning
    titanic_data.drop("Passengerid",axis=1,inplace=True)

    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of sex column after removing one field")
    sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(sex.head(5))
    


    print("Values of Pclass column after removing one field ")
    Pclass=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))

    print("Values of data set after concatenating new columns")
    titanic_data=pd.concat([titanic_data,sex,Pclass],axis=1)
    print(titanic_data.head(5))

    print("Values of data set after removing irrelevent columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5))

    x=titanic_data.drop("Survived",axis=1)
    y=titanic_data["Survived"]
    x.columns=x.columns.astype(str)

    #Step 4 : Data Traning
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5) 

    logmodel=LogisticRegression()

    logmodel.fit(xtrain,ytrain)

    #Step 5 : Data Testing 
    prediction=logmodel.predict(xtest)

    #Step 6 : Calculate Accuracy
    print("Classification report of logistic regression is :")
    print(classification_report(ytest,prediction))

    print("COnfusion Matrix Of Logistic Regression is :")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of logistic regression is :")
    print(accuracy_score(ytest,prediction))


def main():
    print("Supervised Machine learninig")
    print("Logistic Regression on Titanic data set")

    TitanicLogistic()

if __name__=="__main__":
    main()

