import math
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as countplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LogisticRegression

def TitanicLogistic():
    #STep1: Load data
    titanic_data=pd.read_csv('Titanic-Train-Data.csv')
    print("-" * 50)
    #step 2 : Analyze data
    print("Visualisation : Survived and non survived passangers ")
    figure()
    target = "Survived"
    print("-" * 50)
    countplot(data=titanic_data,x=target).set_title("Survived and non survived passangers")
    show()
    print("-" * 50)
    print("Visualisation : Survived and non survived passangers based on Gender")
    figure()
    target = "Survived"
    print("-" * 50)
    countplot(data=titanic_data,x=target,hue="sex").set_title("Survived and non survived passangers based on Gender")
    show()
    print("-" * 50)
    print("Visualisation : Survived and non survived passangers based on passanger class")
    figure()
    target = "Survived"
    print("-" * 50)
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and non survived passangers based on passanger class")
    show()
    print("-" * 50)
    print("Visualisation : Survived and non survived passangers based on age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non survived passangers based on age")
    show()
    print("-" * 50)
    print("Visualisation : Survived and non survived passangers based on Fare")
    figure()
    target = "Survived"
    titanic_data["Fare"].plot.hist().set_title("Visualisation : Survived and non survived passangers based on Fare")
    show()
    print("-" * 50)
    #step3: cleaning data
    titanic_data.drop("zero",axis=1,inplace=True)
    Sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    titanic_data=pd.concat([titanic_data,Sex,Pclass],axis=1)
    titanic_data.drop(["Sex","sibsp","Parc","Embarked"],axis=1,inplace=True)

    x=titanic_data.drop("Survived",axis=1)
    y=titanic_data["Survived"]

    #Step 4 : data training
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.5)
    logmodel=LogisticRegression()

    logmodel.fit(xtrain,ytrain)

    #step 5: data testing
    prediction=logmodel.predict(xtest)
    print("-" * 50)
    # step 6:Calculate accuracy
    print("Classification report of Logistic Regression :")
    print(classification_report(ytest,prediction))
    print("-" * 50)
    print("Confusion matrix of Logistic Regression :")
    print(confusion_matrix(ytest,prediction))
    print("-" * 50)
    print("Accuracy of Logistic Regression : ")
    print(accuracy_score(ytest,prediction))
    print("-" * 50)

def main():

    print("Supervised machine learning")
    print("-"*50)
    print("Logistic regression on Titanic data set")
    print("-" * 50)

    TitanicLogistic()

if __name__=="__main__":
    main()