import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers


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


# Build the ANN model
ann_model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


# Compile the model
ann_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


# Train the model
history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, verbose=1)


# Plot training and validation accuracy and loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluate the model on the test set
test_loss, test_accuracy = ann_model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.2f}")


# Predict on test set
y_test_pred = (ann_model.predict(X_test) > 0.5).astype(int)
print("\nTest Classification Report:")
print(classification_report(y_test, y_test_pred))
