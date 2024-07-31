from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load the California housing dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2, random_state=42)

# Create model instance for regression
model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror')

# Fit model
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, preds)
print(f'Mean Squared Error: {mse:.2f}')