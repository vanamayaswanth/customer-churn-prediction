import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Handling Missing Data (if any)
    data.dropna(inplace=True)  # Remove rows with missing values

    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    data['Location'] = label_encoder.fit_transform(data['Location'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    # Split the data into features (X) and the target variable (y)
    X = data.drop(columns=['CustomerID', 'Name', 'Churn'])
    y = data['Churn']

    # Feature Scaling
    scaler = StandardScaler()
    X[['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']] = scaler.fit_transform(
        X[['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']]
    )

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
