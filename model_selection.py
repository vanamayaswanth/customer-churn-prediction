from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def select_features(X, y, N=5, random_state=42):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=random_state)

    # Fit the model to the data to get feature importances
    rf_classifier.fit(X, y)

    # Get feature importances and map them to feature names
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

    # Sort features by importance in descending order
    feature_importances = feature_importances.sort_values(ascending=False)

    # Select the top N important features
    selected_features = feature_importances.head(N).index
    X_selected = X[selected_features]

    return X_selected

def initialize_models(random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state),
        "Support Vector Machine": SVC(random_state=random_state)
    }
    return models
