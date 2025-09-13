import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_url_model():
    """
    Trains and evaluates a Random Forest model for URL phishing detection.
    Saves the trained model and features, along with visualization plots.
    """
    try:
        # Load the dataset
        df = pd.read_csv('url_dataset.csv')
        print("URL dataset loaded successfully.")

        # Define features and target variable
        features_columns = df.columns.drop(['id', 'CLASS_LABEL'])
        X = df[features_columns]
        y = df['CLASS_LABEL']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        # Initialize and train the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Random Forest model trained successfully.")

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix - URL Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix_url.png')
        print("Saved confusion matrix to confusion_matrix_url.png")
        plt.close()

        # Generate and save Feature Importance plot
        importances = model.feature_importances_
        feature_names = features_columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
        plt.title('Top 20 Feature Importance - URL Classifier')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_url.png')
        print("Saved feature importance plot to feature_importance_url.png")
        plt.close()

        # Save the trained model and feature list
        joblib.dump(model, 'url_model.pkl')
        joblib.dump(list(features_columns), 'url_features.pkl')
        print("Trained model and feature list saved to url_model.pkl and url_features.pkl")

    except FileNotFoundError:
        print("Error: url_dataset.csv not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    train_url_model()
