import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import string
from urllib.parse import urlparse
import re
import idna

# --- URL Feature Extraction Function ---
def extract_url_features_for_training(url):
    """Extracts a set of features from a given URL for training."""
    features = {}

    # Feature 1: Number of dots
    features['NumDots'] = url.count('.')

    # Parse the URL
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path if parsed_url.path else ''
    query = parsed_url.query if parsed_url.query else ''

    # Feature 2: Subdomain level
    features['SubdomainLevel'] = len(hostname.split('.')) - 2 if hostname else 0

    # Feature 3: Path level
    features['PathLevel'] = len(path.split('/')) - 1 if path else 0

    # Feature 4: URL Length
    features['UrlLength'] = len(url)

    # Feature 5: Number of dashes
    features['NumDash'] = url.count('-')

    # Feature 6: Number of dashes in hostname
    features['NumDashInHostname'] = hostname.count('-')

    # Feature 7: Presence of @ symbol
    features['AtSymbol'] = 1 if '@' in url else 0

    # Feature 8: Presence of tilde symbol
    features['TildeSymbol'] = 1 if '~' in url else 0

    # Feature 9: Number of underscores
    features['NumUnderscore'] = url.count('_')

    # Feature 10: Number of percent signs
    features['NumPercent'] = url.count('%')

    # Feature 11: Number of query components
    features['NumQueryComponents'] = len(query.split('&')) if query else 0

    # Feature 12: Number of ampersands
    features['NumAmpersand'] = url.count('&')

    # Feature 13: Number of hash symbols
    features['NumHash'] = url.count('#')

    # Feature 14: Number of numeric characters
    features['NumNumericChars'] = sum(c.isdigit() for c in url)

    # Feature 15: Is not HTTPS
    features['NoHttps'] = 0 if url.startswith('https') else 1

    # Feature 16: Has a random-looking string
    features['RandomString'] = 1 if re.search(r'[A-Za-z0-9]{20,}', url) else 0

    # Feature 17: Has IP address in hostname
    features['IpAddress'] = 1 if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', hostname) else 0

    # New Feature: Has Homoglyphs (correctly implemented)
    features['HasHomoglyph'] = 0
    try:
        url_decoded = idna.decode(url)
        if url != url_decoded:
            features['HasHomoglyph'] = 1
    except idna.IDNAError:
        features['HasHomoglyph'] = 1 # Treat IDNA errors as a sign of homoglyphs

    # New Feature: URL Contains Suspicious Keywords
    suspicious_keywords = ['login', 'signin', 'account', 'verify', 'secure', 'bank', 'update', 'confirm', 'password']
    features['HasSuspiciousKeywords'] = 0
    for keyword in suspicious_keywords:
        if keyword in url.lower():
            features['HasSuspiciousKeywords'] = 1
            break
            
    # New Feature: Punycode Detection
    features['IsPunycode'] = 1 if 'xn--' in url.lower() else 0

    return features

# --- Train URL Model using pre-extracted features ---
def train_url_model(dataset_path):
    """
    Trains and saves the URL classification model using a pre-extracted feature dataset.
    """
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found.")
        return

    # Add a few highly malicious URLs to the training data to force the model to learn
    # The '1' signifies a phishing URL
    malicious_data = [
        ['http://googIe.com-login.secureverify.ru', 1],
        ['https://paypal.com.verify-user-account.ru', 1],
        ['https://apple.com.login.verify-id123.biz', 1],
        ['http://free-gift.cards/claim-now', 1],
    ]
    
    # Extract features from the new malicious data
    malicious_df = pd.DataFrame([extract_url_features_for_training(url) for url, _ in malicious_data])
    malicious_df['CLASS_LABEL'] = [label for _, label in malicious_data]
    
    # Merge with the existing dataset
    df = pd.concat([df, malicious_df], ignore_index=True)
    
    # Drop the 'id' and 'CLASS_LABEL' columns to get the feature matrix
    X = df.drop(columns=['id', 'CLASS_LABEL'], errors='ignore')
    y = df['CLASS_LABEL']

    # We must save the columns to ensure our app.py uses the same feature order
    feature_columns = list(X.columns)

    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    print("Training URL Classification Model...")
    url_model = RandomForestClassifier(n_estimators=100, random_state=42)
    url_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = url_model.predict(X_test)
    print("\n--- URL Model Evaluation ---")
    print(classification_report(y_test, y_pred))

    # Save the model and feature list
    joblib.dump(url_model, 'url_model.pkl')
    joblib.dump(feature_columns, 'url_features.pkl')
    print("URL model and features saved as 'url_model.pkl' and 'url_features.pkl'")

# --- Train SMS Model (unchanged) ---
def train_sms_model():
    """
    Trains and saves the SMS classification model using a dummy dataset based on research.
    """
    data = {
        'v1': ['ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam'],
        'v2': [
            'Hey, what time are we meeting?',
            'Your package is on its way. Use code 345.',
            'See you later, friend.',
            'Lunch tomorrow?',
            'FREE MONEY! Click now to claim your prize.',
            'Congratulations you have won a free iPhone! Click on the link now.',
            'Urgent: Your account is locked. Verify at [fake-link.com]',
            'You won a lottery prize. Send your bank details to claim.'
        ]
    }
    df_sms = pd.DataFrame(data)

    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    df_sms['v2'] = df_sms['v2'].apply(preprocess_text)
    
    print("Training SMS Classification Model...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_sms['v2'])
    y = df_sms['v1']

    sms_model = MultinomialNB()
    sms_model.fit(X, y)
    
    joblib.dump(sms_model, 'sms_model.pkl')
    joblib.dump(vectorizer, 'sms_vectorizer.pkl')
    print("SMS model and vectorizer saved as 'sms_model.pkl' and 'sms_vectorizer.pkl'")

if __name__ == '__main__':
    train_url_model('url_dataset.csv')
    train_sms_model()



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import joblib
# import string
# from urllib.parse import urlparse
# import re

# # --- URL Feature Extraction Function ---
# def extract_url_features(url):
#     """Extracts a set of features from a given URL."""
#     features = {}
    
#     # Feature 1: Number of dots
#     features['NumDots'] = url.count('.')
    
#     # Feature 2: Subdomain level
#     hostname = urlparse(url).hostname
#     features['SubdomainLevel'] = len(hostname.split('.')) - 2 if hostname else 0
    
#     # Feature 3: Path level
#     path = urlparse(url).path
#     features['PathLevel'] = len(path.split('/')) - 1 if path else 0
    
#     # Feature 4: URL Length
#     features['UrlLength'] = len(url)
    
#     # Feature 5: Number of dashes
#     features['NumDash'] = url.count('-')
    
#     # Feature 6: Number of dashes in hostname
#     features['NumDashInHostname'] = hostname.count('-') if hostname else 0
    
#     # Feature 7: Presence of @ symbol
#     features['AtSymbol'] = 1 if '@' in url else 0
    
#     # Feature 8: Presence of tilde symbol
#     features['TildeSymbol'] = 1 if '~' in url else 0
    
#     # Feature 9: Number of underscores
#     features['NumUnderscore'] = url.count('_')
    
#     # Feature 10: Number of percent signs
#     features['NumPercent'] = url.count('%')
    
#     # Feature 11: Number of query components
#     features['NumQueryComponents'] = len(urlparse(url).query.split('&')) if urlparse(url).query else 0
    
#     # Feature 12: Number of ampersands
#     features['NumAmpersand'] = url.count('&')
    
#     # Feature 13: Number of hash symbols
#     features['NumHash'] = url.count('#')
    
#     # Feature 14: Number of numeric characters
#     features['NumNumericChars'] = sum(c.isdigit() for c in url)
    
#     # Feature 15: Is not HTTPS
#     features['NoHttps'] = 0 if url.startswith('https') else 1
    
#     # Feature 16: Has a random-looking string
#     features['RandomString'] = 1 if re.search(r'[A-Za-z0-9]{20,}', url) else 0
    
#     # Feature 17: Has IP address in hostname
#     features['IpAddress'] = 1 if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', hostname) else 0
    
#     # New Feature: Has Homoglyphs
#     homoglyphs_dict = {'o': '0', 'l': '1', 'i': '1', 's': '5', 'a': '@', 'e': '3', 'g': '9', 't': '+', 'z': '2'}
#     features['HasHomoglyph'] = 0
#     for char, homoglyph in homoglyphs_dict.items():
#         if homoglyph in url.lower():
#             features['HasHomoglyph'] = 1
#             break
            
#     # New Feature: URL Contains Suspicious Keywords
#     suspicious_keywords = ['login', 'signin', 'account', 'verify', 'secure', 'bank', 'update', 'confirm', 'password']
#     features['HasSuspiciousKeywords'] = 0
#     for keyword in suspicious_keywords:
#         if keyword in url.lower():
#             features['HasSuspiciousKeywords'] = 1
#             break

#     # New Feature: Punycode Detection
#     features['IsPunycode'] = 1 if 'xn--' in url.lower() else 0

#     return features

# # --- Train URL Model ---
# def train_url_model(dataset_path):
#     """
#     Trains and saves the URL classification model.
#     """
#     try:
#         df = pd.read_csv(dataset_path)
#     except FileNotFoundError:
#         print(f"Error: The file '{dataset_path}' was not found.")
#         return

#     # Extract features from the URL column
#     df['features'] = df['url'].apply(extract_url_features)
    
#     # Create a DataFrame from the extracted features
#     features_df = pd.DataFrame(df['features'].tolist())

#     X = features_df
#     y = df['CLASS_LABEL']

#     # Splitting data for training and testing
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train Random Forest Model
#     print("Training URL Classification Model...")
#     url_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     url_model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = url_model.predict(X_test)
#     print("\n--- URL Model Evaluation ---")
#     print(classification_report(y_test, y_pred))

#     # Save the model and feature list
#     joblib.dump(url_model, 'url_model.pkl')
#     joblib.dump(list(X.columns), 'url_features.pkl')
#     print("URL model and features saved as 'url_model.pkl' and 'url_features.pkl'")

# # --- Train SMS Model ---
# def train_sms_model():
#     """
#     Trains and saves the SMS classification model using a dummy dataset based on research.
#     """
#     # Create a dummy SMS dataset based on the research material.
#     data = {
#         'v1': ['ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam'],
#         'v2': [
#             'Hey, what time are we meeting?',
#             'Your package is on its way. Use code 345.',
#             'See you later, friend.',
#             'Lunch tomorrow?',
#             'FREE MONEY! Click now to claim your prize.',
#             'Congratulations you have won a free iPhone! Click on the link now.',
#             'Urgent: Your account is locked. Verify at [fake-link.com]',
#             'You won a lottery prize. Send your bank details to claim.'
#         ]
#     }
#     df_sms = pd.DataFrame(data)

#     # Preprocessing function
#     def preprocess_text(text):
#         text = text.lower()
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         return text

#     # Apply preprocessing
#     df_sms['v2'] = df_sms['v2'].apply(preprocess_text)

#     # Feature extraction using CountVectorizer
#     print("Training SMS Classification Model...")
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(df_sms['v2'])
#     y = df_sms['v1']

#     # Train Multinomial Naive Bayes Model
#     sms_model = MultinomialNB()
#     sms_model.fit(X, y)
    
#     # Save the model and vectorizer
#     joblib.dump(sms_model, 'sms_model.pkl')
#     joblib.dump(vectorizer, 'sms_vectorizer.pkl')
#     print("SMS model and vectorizer saved as 'sms_model.pkl' and 'sms_vectorizer.pkl'")

# if __name__ == '__main__':
#     train_url_model('url_dataset.csv')
#     train_sms_model()




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, f1_score
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import joblib
# import string
# import re
# import nltk

# # You might need to run this command once to download the stopwords data
# # try:
# #     nltk.data.find('corpora/stopwords')
# # except nltk.downloader.DownloadError:
# #     nltk.download('stopwords')
# from nltk.corpus import stopwords

# def train_url_model(dataset_path):
#     """
#     Trains and saves the URL classification model.
#     """
#     try:
#         df = pd.read_csv(dataset_path)
#     except FileNotFoundError:
#         print(f"Error: The file '{dataset_path}' was not found.")
#         return

#     # Assuming 'CLASS_LABEL' is the target variable
#     # This list must match the columns in your CSV, excluding 'id' and 'CLASS_LABEL'
#     features = [
#         'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 
#         'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore', 
#         'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash', 
#         'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress', 
#         'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname', 
#         'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath', 
#         'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks', 
#         'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms', 
#         'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction', 
#         'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch', 
#         'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow', 
#         'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle', 
#         'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT', 
#         'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 
#         'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
#     ]
#     X = df[features]
#     y = df['CLASS_LABEL']

#     # Splitting data for training and testing
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train Random Forest Model
#     print("Training URL Classification Model...")
#     url_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     url_model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = url_model.predict(X_test)
#     print("\n--- URL Model Evaluation ---")
#     print(classification_report(y_test, y_pred, zero_division=0))

#     # Save the model and feature list
#     joblib.dump(url_model, 'url_model.pkl')
#     joblib.dump(features, 'url_features.pkl')
#     print("URL model and features saved as 'url_model.pkl' and 'url_features.pkl'")

# def train_sms_model():
#     """
#     Trains and saves the SMS classification model using a dummy dataset based on research.
#     """
#     # Create a dummy SMS dataset based on the research material.
#     data = {
#         'v1': ['ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam'],
#         'v2': [
#             'Go until jurong point, crazy..',
#             'I have to be back in the office by 3 PM.',
#             'What are you doing today?',
#             'Happy to hear that! Will call you later.',
#             'URGENT! You have won a prize. Claim it now!',
#             'Your account is locked. Reply "UNLOCK" to reset.',
#             'FREE GADGET! Click here for a chance to win.',
#             'You have a message from your bank. Click link.'
#         ]
#     }
#     df_sms = pd.DataFrame(data)

#     # Preprocessing function
#     def preprocess_text(text):
#         text = text.lower()
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         return text

#     # Apply preprocessing
#     df_sms['v2'] = df_sms['v2'].apply(preprocess_text)

#     # Feature extraction using CountVectorizer
#     print("Training SMS Classification Model...")
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(df_sms['v2'])
#     y = df_sms['v1']

#     # Train Multinomial Naive Bayes Model
#     sms_model = MultinomialNB()
#     sms_model.fit(X, y)
    
#     # Save the model and vectorizer
#     joblib.dump(sms_model, 'sms_model.pkl')
#     joblib.dump(vectorizer, 'sms_vectorizer.pkl')
#     print("SMS model and vectorizer saved as 'sms_model.pkl' and 'sms_vectorizer.pkl'")

# if __name__ == '__main__':
#     # Train both models
#     train_url_model('url_dataset.csv')
#     train_sms_model()






# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, f1_score
# # from sklearn.feature_extraction.text import CountVectorizer
# # from sklearn.naive_bayes import MultinomialNB
# # import joblib
# # import string
# # from nltk.corpus import stopwords
# # import re
# # import nltk

# # def train_url_model(dataset_path):
# #     """
# #     Trains and saves the URL classification model.
# #     """
# #     try:
# #         df = pd.read_csv(dataset_path)
# #     except FileNotFoundError:
# #         print(f"Error: The file '{dataset_path}' was not found.")
# #         return

# #     # Assuming 'CLASS_LABEL' is the target variable
# #     features =]
# #     X = df[features]
# #     y = df

# #     # Splitting data for training and testing
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #     # Train Random Forest Model
# #     print("Training URL Classification Model...")
# #     url_model = RandomForestClassifier(n_estimators=100, random_state=42)
# #     url_model.fit(X_train, y_train)

# #     # Evaluate the model
# #     y_pred = url_model.predict(X_test)
# #     print("\n--- URL Model Evaluation ---")
# #     print(classification_report(y_test, y_pred))

# #     # Save the model and feature list
# #     joblib.dump(url_model, 'url_model.pkl')
# #     joblib.dump(features, 'url_features.pkl')
# #     print("URL model and features saved as 'url_model.pkl' and 'url_features.pkl'")
    
# # def train_sms_model():
# #     """
# #     Trains and saves the SMS classification model using a dummy dataset based on research.
# #     """
# #     # Create a dummy SMS dataset based on the research material.[1, 2]
# #     # This simulates a standard SMS spam collection dataset.
# #     data = {
# #         'v1': ['ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'spam'],
# #         'v2':
# #     }
# #     df_sms = pd.DataFrame(data)

# #     # Preprocessing function
# #     def preprocess_text(text):
# #         text = text.lower()
# #         text = text.translate(str.maketrans('', '', string.punctuation))
# #         return text

# #     # Apply preprocessing
# #     df_sms['v2'] = df_sms['v2'].apply(preprocess_text)

# #     # Feature extraction using CountVectorizer
# #     print("Training SMS Classification Model...")
# #     vectorizer = CountVectorizer()
# #     X = vectorizer.fit_transform(df_sms['v2'])
# #     y = df_sms['v1']

# #     # Train Multinomial Naive Bayes Model
# #     sms_model = MultinomialNB()
# #     sms_model.fit(X, y)
    
# #     # Save the model and vectorizer
# #     joblib.dump(sms_model, 'sms_model.pkl')
# #     joblib.dump(vectorizer, 'sms_vectorizer.pkl')
# #     print("SMS model and vectorizer saved as 'sms_model.pkl' and 'sms_vectorizer.pkl'")

# # if __name__ == '__main__':
# #     # Train both models
# #     train_url_model('url_dataset.csv')
# #     train_sms_model()