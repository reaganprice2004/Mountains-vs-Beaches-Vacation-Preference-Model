# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
from google.colab import files
FILENAME = "mountains_vs_beaches.csv"

try:
   uploaded = files.upload_file(FILENAME)  # Use this to upload the CSV file in Colab
except ValueError:
   #The uploaded was skipped or invalid.
   print("Upload skipped or invalid, continuing...")

# Load the CSV file
data = pd.read_csv(FILENAME)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values Count:\n", missing_values)


# Plot distributions of numeric features
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
   plt.subplot(3, 3, i)
   sns.histplot(data[feature], kde=True)
   plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Select input features based on correlation and logical reasoning
selected_features = ['Proximity_to_Mountains', 'Proximity_to_Beaches', 'Vacation_Budget']
X = data[selected_features]
y = data['Preference']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Justification: Standardization is chosen to center and scale features due to assumptions in logistic regression.
print("Features after Standardization:\n", X_scaled[:5])

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print sizes of each split
print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on validation set to fine-tune model (optional)
y_val_pred = model.predict(X_val)
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Predict on test set and evaluate model
y_test_pred = model.predict(X_test)
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))