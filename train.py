import pandas as pd
import numpy as np
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class MentalHealthAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_data(self):
        """Load the mental health dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Display original class distribution
        print("\nOriginal class distribution:")
        print(self.df['status'].value_counts())
        print(f"\nTotal number of classes: {self.df['status'].nunique()}")
        
        # Sample maximum 2500 examples per class
        print("\n🔄 Sampling data (max 2500 per class)...")
        sampled_dfs = []
        
        for status in self.df['status'].unique():
            class_df = self.df[self.df['status'] == status]
            if len(class_df) > 2500:
                # Sample 2500 examples randomly
                sampled_class = class_df.sample(n=2500, random_state=42)
                print(f"✓ {status}: Sampled 2500 from {len(class_df)} examples")
            else:
                # Keep all examples if less than 2500
                sampled_class = class_df
                print(f"✓ {status}: Kept all {len(class_df)} examples")
            sampled_dfs.append(sampled_class)
        
        # Combine all sampled dataframes
        self.df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the data
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 New dataset shape: {self.df.shape}")
        print("\nNew class distribution:")
        print(self.df['status'].value_counts())
        
        return self.df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_text(self, text):
        """Advanced preprocessing with lemmatization"""
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\nPreparing data...")
        
        # Apply preprocessing
        self.df['processed_text'] = self.df['statement'].apply(self.preprocess_text)
        
        # Remove empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['encoded_status'] = self.label_encoder.fit_transform(self.df['status'])
        
        print(f"\nLabel mapping:")
        for idx, label in enumerate(self.label_encoder.classes_):
            print(f"{idx}: {label}")
        
        # Vectorize text (reduced features for memory efficiency)
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(self.df['processed_text'])
        y = self.df['encoded_status']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        self.class_weight_dict = dict(zip(np.unique(self.y_train), self.class_weights))
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression(
            multi_class='multinomial',
            class_weight=self.class_weight_dict,
            random_state=42
        )
        
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['logistic_regression'] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate
        self.evaluate_model('logistic_regression')
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        
        # Grid search for best parameters (reduced for faster training)
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15],
            'min_samples_split': [5],
            'min_samples_leaf': [2]
        }
        
        rf = RandomForestClassifier(
            class_weight=self.class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate
        self.evaluate_model('random_forest')
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        # Parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train model
        self.models['xgboost'] = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=25
        )
        
        # Evaluate
        self.evaluate_model('xgboost')
        
    def evaluate_model(self, model_name):
        """Evaluate a trained model"""
        print(f"\nEvaluating {model_name}...")
        
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(self.X_test)
            y_pred = self.models[model_name].predict(dtest)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = self.models[model_name].predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'Mentalhealth/Results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_models(self):
        """Save all trained models and preprocessors"""
        print("\nSaving models...")
        
        # Create Model directory if it doesn't exist
        import os
        os.makedirs('Model', exist_ok=True)
        
        # Save vectorizer
        with open('Mentalhealth/Model/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save label encoder
        with open('Mentalhealth/Model/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save models
        for model_name, model in self.models.items():
            if model_name == 'xgboost':
                model.save_model(f'Mentalhealth/Model/{model_name}_model.json')
            else:
                with open(f'Mentalhealth/Model/{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # Save sampling info
        sampling_info = {
            'max_samples_per_class': 2500,
            'total_samples_used': len(self.df),
            'classes': list(self.label_encoder.classes_),
            'class_distribution': self.df['status'].value_counts().to_dict()
        }
        with open('Mentalhealth/Model/sampling_info.pkl', 'wb') as f:
            pickle.dump(sampling_info, f)
        
        print("All models saved successfully!")
        
    def predict_sentiment(self, text, model_name='logistic_regression'):
        """Predict sentiment for new text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(text_vector)
            prediction = self.models[model_name].predict(dtest)
            prediction = np.argmax(prediction, axis=1)[0]
        else:
            prediction = self.models[model_name].predict(text_vector)[0]
        
        # Get label
        label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probability
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(text_vector)
            proba = self.models[model_name].predict(dtest)[0]
        else:
            proba = self.models[model_name].predict_proba(text_vector)[0]
        
        return label, proba[prediction]

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MentalHealthAnalyzer('Mentalhealth/Data/Mentalhealth.csv')
    
    # Load and prepare data
    analyzer.load_data()
    analyzer.prepare_data()
    
    # Train models
    analyzer.train_logistic_regression()
    analyzer.train_random_forest()
    analyzer.train_xgboost()
    
    # Save models
    analyzer.save_models()
    
    # Test prediction
    test_texts = [
        "I feel anxious and worried about everything",
        "Life is beautiful and I'm grateful for everything",
        "I don't want to live anymore"
    ]
    
    print("\n" + "="*50)
    print("Testing predictions:")
    for text in test_texts:
        print(f"\nText: '{text}'")
        for model_name in analyzer.models.keys():
            label, confidence = analyzer.predict_sentiment(text, model_name)
            print(f"{model_name}: {label} (confidence: {confidence:.3f})")