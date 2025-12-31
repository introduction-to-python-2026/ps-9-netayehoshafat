# 1. Load the dataset
import pandas as pd

df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

# 2. Select features
features = ['PPE',  'DFA']
X = df[features]
y = df['status']

# 3. Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 4. Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Choose a model
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

# 6. Test accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

import joblib

joblib.dump(model, 'Parkinsons.joblib')
