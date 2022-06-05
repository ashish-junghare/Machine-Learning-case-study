import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def Logistic(path):

    df = pd.read_csv(path)
    print("-" * 50)
    print("First few entries of dataset")
    print(df.head())
    print("-" * 50)
    plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
    plt.show()
    print("-" * 50)
    X_train, X_test, Y_train, Y_test = train_test_split(df['age'],df.bought_insurance, train_size=0.5)

    model.fit(X_train,Y_train)

    y_predicted=model.predict(X_test)
    data=model.predict_proba(X_test)
    print("-" * 50)
    print("Prediction : ",data)

    print(classification_report(y_test,y_predicted))

    print(confusion_matrix(y_test, y_predicted))

    print(accuracy_score(y_test, y_predicted))




def main():
    print("-"*50)
    print("---Supervised Machine Learning----")
    File='insurance_data.csv'
    Logistic(File)

if __name__=="__main__":
    main()

