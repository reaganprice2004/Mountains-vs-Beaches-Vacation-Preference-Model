import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# Load dataset
from google.colab import files
FILENAME = "mountains_vs_beaches.csv"


try:
    uploaded = files.upload_file(FILENAME)  # Use this to upload the CSV file in Colab
except ValueError:
    # Either the upload was skipped, or multiple were uploaded.
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


# Combine features and target into a single DataFrame for resampling
df_train = pd.concat([X, y], axis=1)


# Create separate DataFrames for the majority and minority classes
df_majority = df_train[df_train['Preference'] == 0]
df_minority = df_train[df_train['Preference'] == 1]


# Upsample the minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # Sample with replacement
                                 n_samples=len(df_majority),  # Match number of samples in majority class- to balance out our data
                                 random_state=42)  # Reproducible results


# Combine the upsampled minority class with the majority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# Check the class distribution after resampling
print("\nClass distribution after resampling:")
print(df_upsampled['Preference'].value_counts())


# Separate features and target after upsampling
X_upsampled = df_upsampled[selected_features]
y_upsampled = df_upsampled['Preference']


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_upsampled)
print("\nFeatures after Standardization:\n", X_scaled[:5])


# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_upsampled, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Print sizes of each split
print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")


# Define the hypothesis function (sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the cost function with regularization
def compute_cost(X, y, theta, lamb=0.1):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    reg_term = (lamb / (2 * m)) * np.sum(theta[1:]**2)  # Regularization
    return cost + reg_term


# Initialize parameters
theta = np.zeros(X_train.shape[1] + 1)  # Adding an extra for the intercept


# Add an intercept term (column of 1s) to X_train
X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])


# Calculate the cost for the initial theta
initial_cost = compute_cost(X_train_with_intercept, y_train.values, theta)
print(f"Initial cost with initial theta: {initial_cost}")


# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Predict on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


# Calculate predictions and probabilities for validation and test sets
y_val_probs = model.predict_proba(X_val)[:, 1]
y_test_probs = model.predict_proba(X_test)[:, 1]


# Compute log loss (error) for validation and test sets
val_error = log_loss(y_val, y_val_probs)
test_error = log_loss(y_test, y_test_probs)


# Compute accuracy for validation and test sets
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# Create data for plotting
errors = [val_error, test_error]
accuracies = [val_accuracy, test_accuracy]
labels = ['Validation', 'Test']


# Plot accuracy vs. error
plt.figure(figsize=(10, 6))
plt.plot(errors, accuracies, marker='o', linestyle='-', color='blue')
plt.scatter(errors, accuracies, color='red', zorder=5)
for i, label in enumerate(labels):
    plt.text(errors[i], accuracies[i], f'{label}', fontsize=12, ha='center', va='bottom')


plt.xlabel('Loss (Error)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Log Loss')
plt.grid(True)
plt.show()






# Predict on validation set
y_val_pred = model.predict(X_val)
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))


# Predict on test set and evaluate model
y_test_pred = model.predict(X_test)
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))

