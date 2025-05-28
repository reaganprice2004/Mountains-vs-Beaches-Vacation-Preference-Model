# 🏖️🏔️ User Vacation Preference Prediction
**Course:** ISE-221: Intelligent Systems

**Timeline:** October 2024-December 2024
## 🗺️ Overview
This project explores the development of classification models to predict whether individuals prefer vacationing in the **mountains** or at the **beach**,based on features such as
- Age
- Proximity to mountains and beaches
- Vacation budget
We implemented both **logistic regression** and an **artificial neural network (ANN)** to compare their performance and identify key predictors of user preference
## 🎯 Project Objectives
- Build a classification model to predict vacation preferences
- Handle and correct for imbalanced classes in the dataset
- Evaluate model performance using classification metrics
- Compare the performance between the two models
- Understand model behavior and identify areas for improvement
## 🧹 Data Preprocessing
- **Dataset:** 'MvS-Scaled.csv' 
- **Features Selected:**
    - `Proximity_to_Mountains`
    - 'Proximity_to_Beaches'
    - 'Vacation_Budget'
    - 'Age'
  - **Target Variable:** 'Preference' (0 = beach, 1 = mountains).
  ## 📐 Feature Scaling
  - Applied 'StandardScaler' to normalize input features (mean = 0, std = 1).
  ## ⚖️ Handling Imbalanced Data
  - Resampled the minority class (Mountains preference) using upsampling to match the majority class (Beach).
  ## 🤖 Model #1: Logistic Regression
  ### Model Details
  - Trained using scaled features
  - Evaluated using precision, recall, and F1-score, and confusion matrix.
  ### Performance Summary
  |  Metric   | Class 0 (Beach)  | Class 1 (Mountain)  |
  |-----------|------------------|---------------------|
  | Precision | 0.80             | 0.50                |
  | Recall    | 0.89             | 0.33                |
  | F1-score  | 0.84             | 0.39                |
- **Observation:** The model performs significantly better on beach preferences due to initial class imbalance.
## 🤖 Model #2: Artificial Neural Network (ANN)
### Implementation
- Framework: TensorFlow Keras
- Layers:
    - Input: 3 neurons
    - Hidden Layer 1: 64 neurons (ReLU)
    - Hidden Layer 2: 32 neurons (ReLU)
    - Output: 1 neuron (Sigmoid for binary classification)
### Training Details
- Epochs: 50
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Binary crossentropy
## Evaluation
- Trained and validated using resampled data.
- Plotted training/validation accuracy and loss.
- Evaluated using classification report and confusion matrix.
## 📊 Performance Comparison
|  Metric   | Logistic Regression  | ANN    |
|-----------|----------------------|--------|
| Training Loss | Converges steadily  | Decreases and stabilizes |
| Validation Loss | Moderate          | Initially drops, then fluctuates |
| Accurary        | High              | Very high |
| Precision       | Decent            | Superior  |
| F1 Score        | Balanced          | Balanced  |
| Confusion Matrix | Some false positives | Fewer errors overall |
## 🔑 Key Learnings and Reflections
- Logistic Regression is interpretable and fast but struggles with complex patterns.
- ANN captures non-linear relationships more effectively.
- Resampling was essential to overcoming class imbalance.
- Regularization and loss visualization provided key insights into model behavior.
## 💡 Future Work/Ideas
- Hyperparameter tuning for both models.
- Add demographic features for better accuracy.
- Apply cross-validation.
- Try ensemble models.
- Deploy as a simple web app for user input/prediction.
## 🏃 Running the Code
1. Upload 'MvS-Scaled.csv' to your environment.
2. Run the 'Initial Code' for basic exploration and Logistic Regression.
3. Run the 'ANN Code' to train and evaluate the neural network.
4. Use the 'Final Code' for advanced training with visualization and regularization.
## 📦 Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
Install dependencies:
'''bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
## 📚 License
MIT - This project was created for educational purposes as part of the ISE-221 Intelligent Systems course. Free to use and modify with attribution!
