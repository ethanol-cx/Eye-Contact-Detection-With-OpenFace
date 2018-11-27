from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pandas as pd

#reads csv of openface and annotations
df = pd.read_csv('source2.csv')

#drops unnsuccessful (no face detected) rows and randomizes order
df = df.drop(df[df.ix[:,4] < 1].index)
df = shuffle(df)

#splits the data frame into features and class
x = df.iloc[:,5:300]
y = df.iloc[:,-1]

#splits data into a training set and a testing set
train, test, train_class, test_class = train_test_split(x,y,test_size=0.33,random_state=42)

#trains model and gets predictions for test set
gnb = GaussianNB()
gnb.fit(train,train_class)
print("model trained")
preds = gnb.predict(test)

#prints accuracy report for test set
print("accuracy score: " + str(accuracy_score(test_class, preds)))
target_names = ['not looking', 'looking']
print(classification_report(test_class, preds, target_names=target_names))

#dump model to drive
joblib.dump(gnb,'model.joblib')
