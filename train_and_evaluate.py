import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your car dataset (assuming it's in a DataFrame)
data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

x = np.array(list(zip(buying, maint)))  # Select the first two features
y = np.array(cls)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, 'car_classification_model.pkl')
