# bike_pdp.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# Cargar datos
df = pd.read_csv("day.csv")

features = ['instant', 'temp', 'hum', 'windspeed']
target = 'cnt'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

PartialDependenceDisplay.from_estimator(model, X_test, features, kind="average")
plt.tight_layout()
plt.show()
