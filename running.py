import numpy as np
import pandas as pd
df = pd.read_csv('running.csv')
print(df)
y = df['injury']
x = df.drop('injury', axis=1)
print(x)
print(y)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
k = 10
selector = SelectKBest(score_func=chi2, k=k)
X_new= selector.fit_transform(x, y)
# Get the indices of the selected features
selected_features = selector.get_support(indices=True)

# Subset the original data with the selected features
X= x.iloc[:, selected_features]
print(X)
feature_scores = selector.scores_
for i, score in enumerate(feature_scores):
    print(f"Feature {i+1}: {score:.2f}")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(X, y)
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state = 68)
boost = XGBClassifier(max_depth = 2, n_estimators = 30)
boost.fit(X_train, y_train)
Y_pred = boost.predict(X_test)
print(Y_pred)
import pickle
pickle.dump(boost,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))