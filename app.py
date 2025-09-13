import os
import re
import joblib
import pandas as pd
from urllib.parse import urlparse
from tldextract import tldextract
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS  # <-- This import is needed
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# A list of sensitive words to check for in URLs
SENSITIVE_WORDS = ['login', 'account', 'verify', 'update', 'signin', 'bank', 'secure', 'paypal', 'ebay', 'amazon']

# Set the project directory as the base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Flask App Setup ---
app = Flask(__name__, template_folder=BASE_DIR)
CORS(app)  # <-- This line initializes CORS for the app

# --- Model Loading ---
url_model = None
url_features = None
sms_model = None
sms_vectorizer = None

# Add a whitelist of trusted domains (add more as needed)
WHITELIST_DOMAINS = [
    'bankofamerica.com',
    'paypal.com',
    'amazon.com',
    'google.com',
    'apple.com',
    'microsoft.com'
]

def is_whitelisted_domain(url):
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}".lower()
    return domain in WHITELIST_DOMAINS

def load_models():
    """Load the trained models from disk."""
    global url_model, url_features, sms_model, sms_vectorizer
    try:
        url_model = joblib.load(os.path.join(BASE_DIR, 'url_model.pkl'))
        url_features = joblib.load(os.path.join(BASE_DIR, 'url_features.pkl'))
        sms_model = joblib.load(os.path.join(BASE_DIR, 'sms_model.pkl'))
        sms_vectorizer = joblib.load(os.path.join(BASE_DIR, 'sms_vectorizer.pkl'))
        print("Models loaded successfully.")
        return True, "Models loaded successfully."
    except FileNotFoundError as e:
        error_message = f"Error: A required model file was not found. Please ensure all training scripts have been run successfully. Missing file: {e.filename}"
        print(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred while loading the models: {e}"
        print(error_message)
        return False, error_message

# --- Feature Extraction Functions for URL ---
def has_at_symbol(url):
    return 1 if '@' in url else 0

def get_url_length(url):
    return len(url)

def get_num_dots(url):
    return url.count('.')

def has_ip_address(url):
    # Regex to check for IP addresses in the hostname
    return 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', urlparse(url).hostname) else 0

def get_subdomain_level(url):
    ext = tldextract.extract(url)
    if ext.subdomain:
        return len(ext.subdomain.split('.'))
    return 0

def has_http_in_hostname(url):
    return 1 if 'http' in urlparse(url).hostname else 0

def get_hostname_length(url):
    return len(urlparse(url).hostname)

def get_path_level(url):
    path = urlparse(url).path
    if path:
        return len(path.split('/'))
    return 0

def get_path_length(url):
    return len(urlparse(url).path)

def get_query_length(url):
    return len(urlparse(url).query)

def get_num_dash(url):
    return url.count('-')

def get_num_dash_in_hostname(url):
    return urlparse(url).hostname.count('-')

def get_tilde_symbol(url):
    return 1 if '~' in url else 0

def get_num_underscore(url):
    return url.count('_')

def get_num_percent(url):
    return url.count('%')

def get_num_query_components(url):
    return len(urlparse(url).query.split('&')) if urlparse(url).query else 0

def get_num_ampersand(url):
    return url.count('&')

def get_num_hash(url):
    return url.count('#')

def get_num_numeric_chars(url):
    return sum(c.isdigit() for c in url)

def no_https(url):
    return 1 if not url.startswith('https') else 0

def has_random_string(url):
    # Checks for long random-looking strings. This is a heuristic.
    path = urlparse(url).path
    if path:
        parts = re.split(r'[/_.-]', path)
        for part in parts:
            if len(part) > 10 and not re.search(r'[a-zA-Z]', part):
                return 1
    return 0

def get_domain_in_subdomains(url):
    ext = tldextract.extract(url)
    domain = ext.domain
    subdomains = ext.subdomain
    return 1 if domain in subdomains else 0

def get_domain_in_paths(url):
    ext = tldextract.extract(url)
    domain = ext.domain
    path = urlparse(url).path
    return 1 if domain in path else 0

def get_double_slash_in_path(url):
    return 1 if '//' in urlparse(url).path else 0

def get_num_sensitive_words(url):
    count = 0
    parsed_url = urlparse(url)
    all_text = parsed_url.netloc + parsed_url.path + parsed_url.query
    for word in SENSITIVE_WORDS:
        count += all_text.lower().count(word)
    return count

def get_embedded_brand_name(url):
    # A simple check for a brand name in the URL, more complex logic is needed for accurate detection
    return 1 if "google" in url.lower() or "apple" in url.lower() or "microsoft" in url.lower() else 0

def extract_features_from_url(url):
    """Extracts all necessary features from a URL string."""
    features = {
        'NumDots': get_num_dots(url),
        'SubdomainLevel': get_subdomain_level(url),
        'PathLevel': get_path_level(url),
        'UrlLength': get_url_length(url),
        'NumDash': get_num_dash(url),
        'NumDashInHostname': get_num_dash_in_hostname(url),
        'AtSymbol': has_at_symbol(url),
        'TildeSymbol': get_tilde_symbol(url),
        'NumUnderscore': get_num_underscore(url),
        'NumPercent': get_num_percent(url),
        'NumQueryComponents': get_num_query_components(url),
        'NumAmpersand': get_num_ampersand(url),
        'NumHash': get_num_hash(url),
        'NumNumericChars': get_num_numeric_chars(url),
        'NoHttps': no_https(url),
        'RandomString': has_random_string(url),
        'IpAddress': has_ip_address(url),
        'DomainInSubdomains': get_domain_in_subdomains(url),
        'DomainInPaths': get_domain_in_paths(url),
        'HttpsInHostname': has_http_in_hostname(url),
        'HostnameLength': get_hostname_length(url),
        'PathLength': get_path_length(url),
        'QueryLength': get_query_length(url),
        'DoubleSlashInPath': get_double_slash_in_path(url),
        'NumSensitiveWords': get_num_sensitive_words(url),
        'EmbeddedBrandName': get_embedded_brand_name(url),
        'PctExtHyperlinks': -1, # These features need a more complex scraping and analysis.
        'PctExtResourceUrls': -1, # They are set to -1 as a placeholder for now.
        'ExtFavicon': -1,
        'InsecureForms': -1,
        'RelativeFormAction': -1,
        'ExtFormAction': -1,
        'AbnormalFormAction': -1,
        'PctNullSelfRedirectHyperlinks': -1,
        'FrequentDomainNameMismatch': -1,
        'FakeLinkInStatusBar': -1,
        'RightClickDisabled': -1,
        'PopUpWindow': -1,
        'SubmitInfoToEmail': -1,
        'IframeOrFrame': -1,
        'MissingTitle': -1,
        'ImagesOnlyInForm': -1,
        'SubdomainLevelRT': -1,
        'UrlLengthRT': -1,
        'PctExtResourceUrlsRT': -1,
        'AbnormalExtFormActionR': -1,
        'ExtMetaScriptLinkRT': -1,
        'PctExtNullSelfRedirectHyperlinksRT': -1,
    }
    
    # Ensure all required features are present and in the correct order
    feature_values = [features[key] for key in url_features]
    return np.array([feature_values])

# --- Endpoints ---
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/check-url', methods=['POST'])
def check_url():
    if url_model is None or url_features is None:
        return jsonify({"error": "URL model not loaded. Please retrain."}), 500
    
    data = request.json
    url = data.get('url', '')

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        features = extract_features_from_url(url)
        prediction_label = url_model.predict(features)[0]
        prediction_proba = url_model.predict_proba(features)[0]
        
        # Calculate risk score as a percentage
        risk_score = round(prediction_proba[1] * 100, 2)
        
        # Determine the prediction and provide a simple explanation
        if prediction_label == 1:
            prediction = "Phishing"
            explanation = "The URL exhibits characteristics commonly found in phishing sites. Be cautious."
        else:
            prediction = "Legitimate"
            explanation = "The URL appears to be legitimate based on the analysis."
        
        return jsonify({
            "prediction": prediction,
            "risk_score": risk_score,
            "explanation": explanation
        })
    except Exception as e:
        print(f"Error during URL prediction: {e}")
        return jsonify({"error": f"Error analyzing URL: {str(e)}"}), 500

@app.route('/check-sms', methods=['POST'])
def check_sms():
    if sms_model is None or sms_vectorizer is None:
        return jsonify({"error": "SMS model not loaded. Please retrain."}), 500

    data = request.json
    sms = data.get('sms', '')
    
    if not sms:
        return jsonify({"error": "No SMS text provided."}), 400

    try:
        # Vectorize the input SMS text
        sms_vectorized = sms_vectorizer.transform([sms])

        # Make a prediction
        prediction_label = sms_model.predict(sms_vectorized)[0]
        prediction_proba = sms_model.predict_proba(sms_vectorized)[0]
        
        # The labels are 'phishing' and 'safe'. We need to map to indices.
        labels = sms_model.classes_
        phishing_index = list(labels).index('phishing')

        # Calculate risk score as a percentage
        risk_score = round(prediction_proba[phishing_index] * 100, 2)
        
        # Determine the prediction and provide a simple explanation
        if prediction_label == 'phishing':
            prediction = "Phishing"
            explanation = "This message contains language and patterns typical of a phishing scam."
        else:
            prediction = "Safe"
            explanation = "This message appears to be safe."

        return jsonify({
            "prediction": prediction,
            "risk_score": risk_score,
            "explanation": explanation
        })
    except Exception as e:
        print(f"Error during SMS prediction: {e}")
        return jsonify({"error": f"Error analyzing SMS: {str(e)}"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        from train_url_model import train_url_model
        from train_sms_model import train_sms_model

        # Retrain both models and save them
        print("\n--- Starting URL Model Retraining ---")
        train_url_model()
        print("\n--- Starting SMS Model Retraining ---")
        train_sms_model()

        # Reload the models into memory
        success, message = load_models()
        if success:
            return jsonify({"message": "Models retrained and reloaded successfully."}), 200
        else:
            return jsonify({"message": message}), 500

    except ImportError:
        return jsonify({"message": "Training scripts not found. Please ensure train_url_model.py and train_sms_model.py are in the same directory."}), 500
    except Exception as e:
        return jsonify({"message": f"An error occurred during retraining: {str(e)}"}), 500

# Initial model loading
success, message = load_models()
if not success:
    print(message)
    # The app can still run, but API calls will fail until models are available.

if __name__ == '__main__':
    # Initial training if models don't exist
    if not os.path.exists('url_model.pkl') or not os.path.exists('sms_model.pkl'):
        print("Models not found. Performing initial training...")
        try:
            from train_url_model import train_url_model
            from train_sms_model import train_sms_model
            train_url_model()
            train_sms_model()
            success, message = load_models()
            if not success:
                print(f"Failed to load models after initial training: {message}")
        except ImportError:
            print("Training scripts not found. Please ensure train_url_model.py and train_sms_model.py are available.")

    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)







# import os
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import joblib
# import pandas as pd
# from urllib.parse import urlparse, urlunparse
# import re
# import string
# import idna

# # --- Load Models and Vectorizer ---
# try:
#     url_model = joblib.load('url_model.pkl')
#     url_features = joblib.load('url_features.pkl')
#     sms_model = joblib.load('sms_model')
#     sms_vectorizer = joblib.load('sms_vectorizer.pkl')
# except FileNotFoundError:
#     print("Error: Model files not found. Please run 'python train_model.py' first.")
#     os._exit(1)

# app = Flask(__name__)
# CORS(app) # Enable CORS for all routes

# # --- URL Feature Extraction Function ---
# def extract_url_features(url):
#     """Extracts a set of features from a given URL."""
#     features = {}

#     # Feature 1: Number of dots
#     features['NumDots'] = url.count('.')

#     # Parse the URL
#     parsed_url = urlparse(url)
#     hostname = parsed_url.hostname or ""

#     # Feature 2: Subdomain level
#     features['SubdomainLevel'] = len(hostname.split('.')) - 2 if hostname else 0

#     # Feature 3: Path level
#     path = parsed_url.path
#     features['PathLevel'] = len(path.split('/')) - 1 if path else 0

#     # Feature 4: URL Length
#     features['UrlLength'] = len(url)

#     # Feature 5: Number of dashes
#     features['NumDash'] = url.count('-')

#     # Feature 6: Number of dashes in hostname
#     features['NumDashInHostname'] = hostname.count('-')

#     # Feature 7: Presence of @ symbol
#     features['AtSymbol'] = 1 if '@' in url else 0

#     # Feature 8: Presence of tilde symbol
#     features['TildeSymbol'] = 1 if '~' in url else 0

#     # Feature 9: Number of underscores
#     features['NumUnderscore'] = url.count('_')

#     # Feature 10: Number of percent signs
#     features['NumPercent'] = url.count('%')

#     # Feature 11: Number of query components
#     query = parsed_url.query
#     features['NumQueryComponents'] = len(query.split('&')) if query else 0

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

#     # Ensure all features are present in the correct order
#     feature_values = [features[f] for f in url_features]

#     return feature_values

# # --- Main Routes ---
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/check-url', methods=['POST'])
# def check_url():
#     data = request.get_json(silent=True)
#     if not data or 'url' not in data:
#         return jsonify({"error": "No URL provided"}), 400
    
#     url = data['url']

#     try:
#         features = extract_url_features(url)
#         features_df = pd.DataFrame([features], columns=url_features)
        
#         # Predict using the model
#         prediction = url_model.predict(features_df)[0]
#         prediction_proba = url_model.predict_proba(features_df)[0]
#         risk_score = prediction_proba[1] * 100

#         result = {
#             "prediction": "Phishing" if prediction == 1 else "Safe",
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on a combination of URL-based features."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/check-sms', methods=['POST'])
# def check_sms():
#     data = request.get_json(silent=True)
#     if not data or 'sms' not in data:
#         return jsonify({"error": "No SMS text provided"}), 400
    
#     sms = data['sms']
    
#     try:
#         # Preprocess and vectorize the SMS text
#         processed_sms = [sms.lower().translate(str.maketrans('', '', string.punctuation))]
#         sms_vectorized = sms_vectorizer.transform(processed_sms)
        
#         # Predict using the model
#         prediction = sms_model.predict(sms_vectorized)[0]
#         prediction_proba = sms_model.predict_proba(sms_vectorized)[0]
        
#         if prediction == 'spam':
#             risk_score = prediction_proba[1] * 100
#             classification = "Phishing"
#         else:
#             risk_score = (1 - prediction_proba[0]) * 100
#             classification = "Safe"
            
#         result = {
#             "prediction": classification,
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on keywords and text patterns."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# import os
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import joblib
# import pandas as pd
# from urllib.parse import urlparse
# import re

# # --- Load Models and Vectorizer ---
# try:
#     url_model = joblib.load('url_model.pkl')
#     url_features = joblib.load('url_features.pkl')
#     sms_model = joblib.load('sms_model.pkl')
#     sms_vectorizer = joblib.load('sms_vectorizer.pkl')
# except FileNotFoundError:
#     print("Error: Model files not found. Please run 'python train_model.py' first.")
#     # Exit if models are not found to prevent the server from starting with errors
#     os._exit(1)

# app = Flask(__name__)
# CORS(app) # Enable CORS for all routes

# # --- URL Feature Extraction Function ---
# def extract_url_features(url):
#     features = {}
    
#     # Simple features based on the provided dataset
#     features['NumDots'] = url.count('.')
#     features['SubdomainLevel'] = len(urlparse(url).hostname.split('.')) - 2
#     features['PathLevel'] = len(urlparse(url).path.split('/')) - 1
#     features['UrlLength'] = len(url)
#     features['NumDash'] = url.count('-')
#     features['NumDashInHostname'] = urlparse(url).hostname.count('-') if urlparse(url).hostname else 0
#     features['AtSymbol'] = 1 if '@' in url else 0
#     features['TildeSymbol'] = 1 if '~' in url else 0
#     features['NumUnderscore'] = url.count('_')
#     features['NumPercent'] = url.count('%')
#     features['NumQueryComponents'] = len(urlparse(url).query.split('&')) if urlparse(url).query else 0
#     features['NumAmpersand'] = url.count('&')
#     features['NumHash'] = url.count('#')
#     features['NumNumericChars'] = sum(c.isdigit() for c in url)
#     features['NoHttps'] = 0 if url.startswith('https') else 1
#     features['RandomString'] = 1 if re.search(r'[A-Za-z0-9]{20,}', url) else 0
#     features['IpAddress'] = 1 if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', urlparse(url).hostname) else 0
#     features['DomainInSubdomains'] = 0 # Placeholder for simplicity
#     features['DomainInPaths'] = 0 # Placeholder for simplicity
#     features['HttpsInHostname'] = 1 if 'https' in urlparse(url).hostname else 0
#     features['HostnameLength'] = len(urlparse(url).hostname) if urlparse(url).hostname else 0
#     features['PathLength'] = len(urlparse(url).path)
#     features['QueryLength'] = len(urlparse(url).query)
#     features['DoubleSlashInPath'] = 1 if '//' in urlparse(url).path else 0
#     features['NumSensitiveWords'] = 1 if re.search(r'login|signin|account|bank', url, re.IGNORECASE) else 0
#     features['EmbeddedBrandName'] = 0 # Placeholder
#     features['PctExtHyperlinks'] = 0.0 # Placeholder
#     features['PctExtResourceUrls'] = 0.0 # Placeholder
#     features['ExtFavicon'] = 0 # Placeholder
#     features['InsecureForms'] = 0 # Placeholder
#     features['RelativeFormAction'] = 0 # Placeholder
#     features['ExtFormAction'] = 0 # Placeholder
#     features['AbnormalFormAction'] = 0 # Placeholder
#     features['PctNullSelfRedirectHyperlinks'] = 0.0 # Placeholder
#     features['FrequentDomainNameMismatch'] = 0 # Placeholder
#     features['FakeLinkInStatusBar'] = 0 # Placeholder
#     features['RightClickDisabled'] = 0 # Placeholder
#     features['PopUpWindow'] = 0 # Placeholder
#     features['SubmitInfoToEmail'] = 0 # Placeholder
#     features['IframeOrFrame'] = 0 # Placeholder
#     features['MissingTitle'] = 0 # Placeholder
#     features['ImagesOnlyInForm'] = 0 # Placeholder
#     features['SubdomainLevelRT'] = 0 # Placeholder
#     features['UrlLengthRT'] = 0 # Placeholder
#     features['PctExtResourceUrlsRT'] = 0.0 # Placeholder
#     features['AbnormalExtFormActionR'] = 0 # Placeholder
#     features['ExtMetaScriptLinkRT'] = 0 # Placeholder
#     features['PctExtNullSelfRedirectHyperlinksRT'] = 0.0 # Placeholder
    
#     return [features[f] for f in url_features]

# # --- Main Routes ---
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/check-url', methods=['POST'])
# def check_url():
#     data = request.get_json(silent=True)
#     if not data or 'url' not in data:
#         return jsonify({"error": "No URL provided"}), 400
    
#     url = data['url']
#     if not url.startswith('http'):
#         url = 'http://' + url

#     try:
#         features = extract_url_features(url)
#         features_df = pd.DataFrame([features], columns=url_features)
        
#         # Predict using the model
#         prediction = url_model.predict(features_df)[0]
#         prediction_proba = url_model.predict_proba(features_df)[0]
#         risk_score = prediction_proba[1] * 100

#         result = {
#             "prediction": "Phishing" if prediction == 1 else "Safe",
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on a combination of URL-based features."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/check-sms', methods=['POST'])
# def check_sms():
#     data = request.get_json(silent=True)
#     if not data or 'sms' not in data:
#         return jsonify({"error": "No SMS text provided"}), 400
    
#     sms = data['sms']
    
#     try:
#         # Preprocess and vectorize the SMS text
#         processed_sms = [sms.lower().translate(str.maketrans('', '', string.punctuation))]
#         sms_vectorized = sms_vectorizer.transform(processed_sms)
        
#         # Predict using the model
#         prediction = sms_model.predict(sms_vectorized)[0]
#         prediction_proba = sms_model.predict_proba(sms_vectorized)[0]
        
#         if prediction == 'spam':
#             risk_score = prediction_proba[1] * 100
#             classification = "Phishing"
#         else:
#             risk_score = (1 - prediction_proba[0]) * 100
#             classification = "Safe"
            
#         result = {
#             "prediction": classification,
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on keywords and text patterns."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)





# import os
# from flask import Flask, request, jsonify, render_template
# import joblib
# import pandas as pd
# from urllib.parse import urlparse
# import re

# # --- Load Models and Vectorizer ---
# try:
#     url_model = joblib.load('url_model.pkl')
#     url_features = joblib.load('url_features.pkl')
#     sms_model = joblib.load('sms_model.pkl')
#     sms_vectorizer = joblib.load('sms_vectorizer.pkl')
# except FileNotFoundError:
#     print("Error: Model files not found. Please run 'python train_model.py' first.")
#     # Exit if models are not found to prevent the server from starting with errors
#     os._exit(1)

# app = Flask(__name__)

# # --- URL Feature Extraction Function ---
# def extract_url_features(url):
#     features = {}
    
#     # Simple features based on the provided dataset
#     features['NumDots'] = url.count('.')
#     features['SubdomainLevel'] = len(urlparse(url).hostname.split('.')) - 2
#     features['PathLevel'] = len(urlparse(url).path.split('/')) - 1
#     features['UrlLength'] = len(url)
#     features['NumDash'] = url.count('-')
#     features['NumDashInHostname'] = urlparse(url).hostname.count('-') if urlparse(url).hostname else 0
#     features['AtSymbol'] = 1 if '@' in url else 0
#     features['TildeSymbol'] = 1 if '~' in url else 0
#     features['NumUnderscore'] = url.count('_')
#     features['NumPercent'] = url.count('%')
#     features['NumQueryComponents'] = len(urlparse(url).query.split('&')) if urlparse(url).query else 0
#     features['NumAmpersand'] = url.count('&')
#     features['NumHash'] = url.count('#')
#     features['NumNumericChars'] = sum(c.isdigit() for c in url)
#     features['NoHttps'] = 0 if url.startswith('https') else 1
#     features['RandomString'] = 1 if re.search(r'[A-Za-z0-9]{20,}', url) else 0
#     features['IpAddress'] = 1 if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', urlparse(url).hostname) else 0
#     features['DomainInSubdomains'] = 0 # Placeholder for simplicity
#     features['DomainInPaths'] = 0 # Placeholder for simplicity
#     features['HttpsInHostname'] = 1 if 'https' in urlparse(url).hostname else 0
#     features['HostnameLength'] = len(urlparse(url).hostname) if urlparse(url).hostname else 0
#     features['PathLength'] = len(urlparse(url).path)
#     features['QueryLength'] = len(urlparse(url).query)
#     features['DoubleSlashInPath'] = 1 if '//' in urlparse(url).path else 0
#     features['NumSensitiveWords'] = 1 if re.search(r'login|signin|account|bank', url, re.IGNORECASE) else 0
#     features['EmbeddedBrandName'] = 0 # Placeholder
#     features['PctExtHyperlinks'] = 0.0 # Placeholder
#     features['PctExtResourceUrls'] = 0.0 # Placeholder
#     features['ExtFavicon'] = 0 # Placeholder
#     features['InsecureForms'] = 0 # Placeholder
#     features['RelativeFormAction'] = 0 # Placeholder
#     features['ExtFormAction'] = 0 # Placeholder
#     features['AbnormalFormAction'] = 0 # Placeholder
#     features['PctNullSelfRedirectHyperlinks'] = 0.0 # Placeholder
#     features['FrequentDomainNameMismatch'] = 0 # Placeholder
#     features['FakeLinkInStatusBar'] = 0 # Placeholder
#     features['RightClickDisabled'] = 0 # Placeholder
#     features['PopUpWindow'] = 0 # Placeholder
#     features['SubmitInfoToEmail'] = 0 # Placeholder
#     features['IframeOrFrame'] = 0 # Placeholder
#     features['MissingTitle'] = 0 # Placeholder
#     features['ImagesOnlyInForm'] = 0 # Placeholder
#     features['SubdomainLevelRT'] = 0 # Placeholder
#     features['UrlLengthRT'] = 0 # Placeholder
#     features['PctExtResourceUrlsRT'] = 0.0 # Placeholder
#     features['AbnormalExtFormActionR'] = 0 # Placeholder
#     features['ExtMetaScriptLinkRT'] = 0 # Placeholder
#     features['PctExtNullSelfRedirectHyperlinksRT'] = 0.0 # Placeholder
    
#     return [features[f] for f in url_features]

# # --- Main Routes ---
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/check-url', methods=['POST'])
# def check_url():
#     data = request.get_json(silent=True)
#     if not data or 'url' not in data:
#         return jsonify({"error": "No URL provided"}), 400
    
#     url = data['url']
#     if not url.startswith('http'):
#         url = 'http://' + url

#     try:
#         features = extract_url_features(url)
#         features_df = pd.DataFrame([features], columns=url_features)
        
#         # Predict using the model
#         prediction = url_model.predict(features_df)[0]
#         prediction_proba = url_model.predict_proba(features_df)[0]
#         risk_score = prediction_proba[1] * 100

#         result = {
#             "prediction": "Phishing" if prediction == 1 else "Safe",
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on a combination of URL-based features."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/check-sms', methods=['POST'])
# def check_sms():
#     data = request.get_json(silent=True)
#     if not data or 'sms' not in data:
#         return jsonify({"error": "No SMS text provided"}), 400
    
#     sms = data['sms']
    
#     try:
#         # Preprocess and vectorize the SMS text
#         processed_sms = [sms.lower().translate(str.maketrans('', '', string.punctuation))]
#         sms_vectorized = sms_vectorizer.transform(processed_sms)
        
#         # Predict using the model
#         prediction = sms_model.predict(sms_vectorized)[0]
#         prediction_proba = sms_model.predict_proba(sms_vectorized)[0]
        
#         if prediction == 'spam':
#             risk_score = prediction_proba[1] * 100
#             classification = "Phishing"
#         else:
#             risk_score = (1 - prediction_proba[0]) * 100
#             classification = "Safe"
            
#         result = {
#             "prediction": classification,
#             "risk_score": round(risk_score, 2),
#             "explanation": "Based on keywords and text patterns."
#         }
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, request, jsonify, render_template
# import joblib
# import pandas as pd
# import string
# import re
# import math
# from urllib.parse import urlparse

# app = Flask(__name__, static_folder='static', template_folder='templates')

# # Load the pre-trained models and other necessary data
# try:
#     url_model = joblib.load('url_model.pkl')
#     url_features = joblib.load('url_features.pkl')
#     sms_model = joblib.load('sms_model.pkl')
#     sms_vectorizer = joblib.load('sms_vectorizer.pkl')
# except FileNotFoundError as e:
#     print(f"Error loading model files: {e}. Please run `python train_model.py` first.")
#     url_model = None
#     sms_model = None
#     url_features =
#     sms_vectorizer = None

# # --- SMS Feature Extraction & Preprocessing ---
# def preprocess_sms(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     return text

# # --- URL Feature Extraction Function ---
# def extract_url_features(url):
#     """
#     A simple feature extraction function for a given URL string.
#     This function extracts some of the features present in your dataset.
#     """
#     features = {}
    
#     # NumDots: Number of dots in the URL
#     features = url.count('.')
    
#     # UrlLength: Length of the URL string
#     features['UrlLength'] = len(url)
    
#     # NumDash: Number of hyphens in the URL
#     features = url.count('-')

#     # AtSymbol: Presence of '@'
#     features = 1 if '@' in url else 0

#     # NoHttps: Absence of 'https://'
#     features['NoHttps'] = 1 if 'https://' not in url.lower() else 0
    
#     # IpAddress: Check for IP address in hostname
#     parsed_url = urlparse(url)
#     features['IpAddress'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.hostname) else 0

#     # The remaining features in your dataset are complex.
#     # For a fully functional prototype, we will set them to default values.
#     # You can expand this function later to include more complex logic.
#     default_features = {
#         'SubdomainLevel': 0, 'PathLevel': 0, 'NumDashInHostname': 0, 'TildeSymbol': 0, 'NumUnderscore': 0,
#         'NumPercent': 0, 'NumQueryComponents': 0, 'NumAmpersand': 0, 'NumHash': 0, 'NumNumericChars': 0,
#         'RandomString': 0, 'DomainInSubdomains': 0, 'DomainInPaths': 0, 'HttpsInHostname': 0,
#         'HostnameLength': 0, 'PathLength': 0, 'QueryLength': 0, 'DoubleSlashInPath': 0, 'NumSensitiveWords': 0,
#         'EmbeddedBrandName': 0, 'PctExtHyperlinks': 0.0, 'PctExtResourceUrls': 0.0, 'ExtFavicon': 0,
#         'InsecureForms': 0, 'RelativeFormAction': 0, 'ExtFormAction': 0, 'AbnormalFormAction': 0,
#         'PctNullSelfRedirectHyperlinks': 0.0, 'FrequentDomainNameMismatch': 0, 'FakeLinkInStatusBar': 0,
#         'RightClickDisabled': 0, 'PopUpWindow': 0, 'SubmitInfoToEmail': 0, 'IframeOrFrame': 0,
#         'MissingTitle': 0, 'ImagesOnlyInForm': 0, 'SubdomainLevelRT': 0, 'UrlLengthRT': 0,
#         'PctExtResourceUrlsRT': 0, 'AbnormalExtFormActionR': 0, 'ExtMetaScriptLinkRT': 0,
#         'PctExtNullSelfRedirectHyperlinksRT': 0
#     }
#     features.update(default_features)

#     # Return the feature vector in the same order as the training data
#     return [features[f] for f in url_features]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/classify-url', methods=)
# def classify_url():
#     if not url_model or not url_features:
#         return jsonify({'error': 'URL model not loaded. Please run train_model.py'}), 500
    
#     data = request.json
#     url = data.get('url')
#     if not url:
#         return jsonify({'error': 'URL not provided'}), 400
    
#     try:
#         # Extract features from the input URL
#         features = extract_url_features(url)
        
#         # Predict the class
#         prediction = url_model.predict([features])
#         prediction_proba = url_model.predict_proba([features])
        
#         # Format the response
#         result = {
#             'classification': 'phishing' if prediction == 1 else 'legitimate',
#             'score': float(prediction_proba[1])
#         }
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/classify-sms', methods=)
# def classify_sms():
#     if not sms_model or not sms_vectorizer:
#         return jsonify({'error': 'SMS model not loaded. Please run train_model.py'}), 500

#     data = request.json
#     message = data.get('message')
#     if not message:
#         return jsonify({'error': 'Message not provided'}), 400

#     try:
#         # Preprocess and vectorize the input message
#         preprocessed_message = preprocess_sms(message)
#         vectorized_message = sms_vectorizer.transform([preprocessed_message])
        
#         # Predict the class and probabilities
#         prediction = sms_model.predict(vectorized_message)
#         prediction_proba = sms_model.predict_proba(vectorized_message)

#         # Get class labels from the model
#         class_labels = list(sms_model.classes_)
#         ham_proba = prediction_proba[class_labels.index('ham')]
#         spam_proba = prediction_proba[class_labels.index('spam')]
        
#         # Format the response
#         result = {
#             'classification': 'spam' if prediction == 'spam' else 'ham',
#             'confidence': {
#                 'ham': float(ham_proba),
#                 'spam': float(spam_proba)
#             }
#         }
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)