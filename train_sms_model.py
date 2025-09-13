import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_sms_model():
    """
    Trains and evaluates a Naive Bayes model for SMS phishing detection.
    Saves the trained model and vectorizer, along with a confusion matrix plot.
    """
    try:
        # Load the dataset
        df = pd.read_csv('sms_dataset.csv')
        print("SMS dataset loaded successfully.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        # Convert text data into TF-IDF features
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        print("Text data vectorized successfully.")

        # Initialize and train the Naive Bayes Classifier
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)
        print("Naive Bayes model trained successfully.")

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test_vectorized)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='phishing', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label='phishing', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='phishing', zero_division=0)

        print("\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=['phishing', 'safe'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Safe'], yticklabels=['Phishing', 'Safe'])
        plt.title('Confusion Matrix - SMS Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix_sms.png')
        print("Saved confusion matrix to confusion_matrix_sms.png")
        plt.close()

        # Save the trained model and vectorizer
        joblib.dump(model, 'sms_model.pkl')
        joblib.dump(vectorizer, 'sms_vectorizer.pkl')
        print("Trained model and vectorizer saved to sms_model.pkl and sms_vectorizer.pkl")

    except FileNotFoundError:
        print("Error: sms_dataset.csv not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    train_sms_model()
