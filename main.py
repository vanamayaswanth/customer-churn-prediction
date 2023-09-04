import data_processing
import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess the data
X, y = data_processing.preprocess_data("customer_churn_large_dataset.xlsx")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = data_processing.split_data(X, y)

# Select top features
X_selected = model_selection.select_features(X_train, y_train)

# Initialize models
models = model_selection.initialize_models()

# Create dictionaries to store the evaluation metrics for each model
accuracy_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}
roc_auc_scores = {}

# Train and evaluate each model
for model_name, model in models.items():
    # Fit the model on the training data
    model.fit(X_selected, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores[model_name] = accuracy_score(y_test, y_pred)
    precision_scores[model_name] = precision_score(y_test, y_pred)
    recall_scores[model_name] = recall_score(y_test, y_pred)
    f1_scores[model_name] = f1_score(y_test, y_pred)
    roc_auc_scores[model_name] = roc_auc_score(y_test, y_pred)

# Print evaluation metrics for each model
print("Model Evaluation Metrics:")
for model_name in models.keys():
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_scores[model_name]:.4f}")
    print(f"Precision: {precision_scores[model_name]:.4f}")
    print(f"Recall: {recall_scores[model_name]:.4f}")
    print(f"F1-Score: {f1_scores[model_name]:.4f}")
    print(f"ROC AUC: {roc_auc_scores[model_name]:.4f}")
    print("\n")
