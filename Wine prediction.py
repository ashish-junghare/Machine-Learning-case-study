from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd

def WinePredictor(csv_file):

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
    File='WinePredictor.csv'
    WinePredictor(File)

if __name__=="__main__":
    main()

