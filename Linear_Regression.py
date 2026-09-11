import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

data = {
    'Hours_studied': [1,2,3,4,5,6,7,8,9,10],
    'Score': [12,45,32,66,78,54,21,89,77, 99]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Hours_studied']]   # needs to be 2D
y = df['Score']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=78)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Model coefficient (slope): {model.coef_[0]}")
print(f"Model intercept: {model.intercept_}")

# Predict score for a new value, e.g. 6.5 hours studied
new_hours = pd.DataFrame({'Hours_studied': [9]})
predicted_score = model.predict(new_hours)
print(f"Predicted score for 9 hours studied: {predicted_score[0]}")

# Plot actual vs predicted
plt.scatter(df['Hours_studied'], y, color='blue', label='Actual data')
plt.plot(df['Hours_studied'], model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score')
plt.legend()
plt.show()


