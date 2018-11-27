import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import hmm

#  K-Nearest-Neighbour Model


def knn_classify(X_train, y_train, X_test, y_test, X, y):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_prediction = knn.predict(X_test)
    print(classification_report(y_test, knn_prediction))
    joblib.dump(knn, 'knn.joblib')

# Logistic Regression Model


def lr_classify(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(
        random_state=0, solver='sag', multi_class='multinomial').fit(X_train, y_train)
    lr_prediction = lr_model.predict(X_test)
    print(classification_report(y_test, lr_prediction))
    joblib.dump(lr_model, 'lr.joblib')

# SVM model


def svm_classify(X_train, y_train, X_test, y_test):
    svc_model = SVC(gamma='auto', kernel='poly',
                    coef0=0.5).fit(X_train, y_train)
    svc_prediction = svc_model.predict(X_test)
    print(classification_report(y_test, svc_prediction))
    joblib.dump(svc_model, 'svm.joblib')

# Decision Tree model


def dt_classify(X_train, y_train, X_test, y_test, X, y):
    dt_model = DecisionTreeClassifier(criterion='gini')
    dt_model.fit(X_train, y_train)
    dt_prediction = dt_model.predict(X_test)
    print("Decision Tree Classifier")
    print(classification_report(y_test, dt_prediction))
    joblib.dump(dt_model, 'dt3.joblib')


def main(argv):

    # load the data
    data = pd.read_csv("source.csv")
    data = data[[' timestamp', ' success', ' gaze_angle_x',
                 ' gaze_angle_y', ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz', 'Annotation']]
    data.columns = ['timestamp', 'success', 'gaze_angle_x',
                    'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz', 'Annotation']
    data = data[data['success'] == 1]

    # extract the following features
    X = data[['gaze_angle_x',
              'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz']]
    y = data['Annotation']
    y = [int(a) for a in y]

    # repeate the random permutation of cross-validation
    for i in range(5):
        # split the train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)
        knn_classify(X_train, y_train, X_test, y_test, X, y)
        dt_classify(X_train, y_train, X_test, y_test, X, y)


if __name__ == "__main__":
    main(sys.argv[1:])
