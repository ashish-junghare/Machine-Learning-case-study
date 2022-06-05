import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def playpredict(csv_file):

    batches = pd.read_csv(csv_file)

    batches_sheet1=pd.read_csv(csv_file,index_col=0)


    le = preprocessing.LabelEncoder()
    batches.Wether=le.fit_transform(batches.Wether)
    batches.Temperature=le.fit_transform(batches.Temperature)
    batches.Play=le.fit_transform(batches.Play)

    features = list(zip(batches.Wether,batches.Temperature))
    X_train,X_test,Y_train,Y_test = train_test_split(features,batches.Play,test_size=0.5)
    classifier=KNeighborsClassifier()
    classifier.fit(X_train,Y_train)


    predictions=classifier.predict(X_test)

    Accuracy=accuracy_score(Y_test,predictions)
    print("Accuracy is : ",Accuracy*100,"%")

def main():
    File='MarvellousInfosystems_PlayPredictor.csv'
    playpredict(File)

if __name__=="__main__":
    main()

