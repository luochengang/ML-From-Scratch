from sklearn import datasets

# Import helper functions
from mlfromscratch.utils import train_test_split, accuracy_score
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import ClassificationTree


def main():
    print ("-- Classification Tree --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # y_test 长度为60的行向量
    # y_train 长度为90的行向量, np.shape(y_train) = 90, len(np.shape(y_train)) = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
