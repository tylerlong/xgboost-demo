from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=0.2, random_state=42)

# Create model instance for multi-class classification
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='multi:softprob')

# Fit model
bst.fit(X_train, y_train)

# Make predictions
preds = bst.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy * 100:.2f}%')
