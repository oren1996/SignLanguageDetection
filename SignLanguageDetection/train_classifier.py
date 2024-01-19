import pickle
import numpy as np

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

f = open('data.pickle', 'rb')
model_dict = pickle.load(f)
f.close()

data = np.array(model_dict['data'])
labels = model_dict['labels']

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, shuffle=True, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(y_test, y_pred)
print(score)

for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(y_test[i], y_pred[i])

f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()


