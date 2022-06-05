#-------Breast cancer case study-----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train,X_test,Y_train,Y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,Y_train)
    training_accuracy.append(clf.score(X_train, Y_train))

    test_accuracy.append(clf.score(X_test,Y_test))

plt.plot(neighbors_settings,training_accuracy,label="Training Accuracy")
plt.plot(neighbors_settings,test_accuracy,label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
