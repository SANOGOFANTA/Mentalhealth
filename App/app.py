import gradio as gr
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class MentalHealthPredictor:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.load_models()
        
    def load_models(self):
        """Load all saved models and preprocessors"""
        try:
            # Load vectorizer
            with open('Model/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load label encoder
            with open('Projet_MLOPs/Model/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load Logistic Regression
            with open('Projet_MLOPs/Model/logistic_regression_model.pkl', 'rb') as f:
                self.models['Logistic Regression'] = pickle.load(f)
            
            # Load Random Forest
            with open('Projet_MLOPs/Model/random_forest_model.pkl', 'rb') as f:
                self.models['Random Forest'] = pickle.load(f)
            
            # Load XGBoost
            self.models['XGBoost'] = xgb.Booster()
            self.models['XGBoost'].load_model('Model/xgboost_model.json')
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure all model files are in the 'Model' directory")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
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
    
    def predict(self, text, model_name):
        """Make prediction for given text and model"""
        if not text:
            return "Please enter some text", {}
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return "Text is empty after preprocessing", {}
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Make prediction based on model
            if model_name == 'XGBoost':
                dtest = xgb.DMatrix(text_vector)
                predictions = self.models[model_name].predict(dtest)
                prediction_idx = np.argmax(predictions, axis=1)[0]
                probabilities = predictions[0]
            else:
                prediction_idx = self.models[model_name].predict(text_vector)[0]
                probabilities = self.models[model_name].predict_proba(text_vector)[0]
            
            # Get label
            predicted_label = self.label_encoder.inverse_transform([prediction_idx])[0]
            
            # Create probability dictionary for all classes
            prob_dict = {}
            for idx, label in enumerate(self.label_encoder.classes_):
                prob_dict[label] = float(probabilities[idx])
            
            # Sort by probability
            prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            
            return predicted_label, prob_dict
            
        except Exception as e:
            return f"Error: {str(e)}", {}
    
    def predict_all_models(self, text):
        """Get predictions from all models"""
        if not text:
            return "Please enter some text", "", {}
        
        results = []
        all_probs = {}
        
        for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            pred_label, probs = self.predict(text, model_name)
            results.append(f"**{model_name}**: {pred_label}")
            
            # Store probabilities for visualization
            if probs:
                all_probs[model_name] = probs
        
        # Create summary
        summary = "\n".join(results)
        
        # Get consensus (most common prediction)
        predictions = [self.predict(text, model)[0] for model in self.models.keys()]
        consensus = max(set(predictions), key=predictions.count)
        
        return summary, f"**Consensus Prediction**: {consensus}", all_probs

# Initialize predictor
predictor = MentalHealthPredictor()

# Gradio interface functions
def single_model_prediction(text, model_name):
    """Interface function for single model prediction"""
    label, probs = predictor.predict(text, model_name)
    return f"Predicted Status: **{label}**", probs

def ensemble_prediction(text):
    """Interface function for ensemble prediction"""
    summary, consensus, _ = predictor.predict_all_models(text)
    return summary + "\n\n" + consensus

def analyze_sentiment(text):
    """Complete analysis with all models"""
    summary, consensus, all_probs = predictor.predict_all_models(text)
    
    # Create detailed output
    output = f"### Analysis Results\n\n{summary}\n\n{consensus}\n\n"
    
    # Add confidence scores
    output += "### Confidence Scores by Model\n\n"
    for model_name, probs in all_probs.items():
        if probs:
            top_pred = max(probs, key=probs.get)
            output += f"**{model_name}**: {top_pred} ({probs[top_pred]:.2%} confidence)\n"
    
    return output

# Example texts
examples = [
    ["I'm feeling really anxious and can't sleep at night"],
    ["Life is wonderful and I'm grateful for everything"],
    ["I don't see any point in continuing anymore"],
    ["Just another normal day at work"],
    ["I'm stressed about my exams but trying to stay positive"],
    ["Everything feels hopeless and dark"],
    ["Feeling a bit down today but I know it will get better"]
]

# Create Gradio interface
with gr.Blocks(title="Mental Health Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Mental Health Sentiment Analysis
    
    This application analyzes text to detect mental health sentiments using machine learning models.
    It can identify various mental states including anxiety, depression, suicidal ideation, and normal states.
    
    **Note**: This is for educational purposes only and should not replace professional mental health assessment.
    """)
    
    with gr.Tab("Single Model Prediction"):
        with gr.Row():
            with gr.Column():
                text_input1 = gr.Textbox(
                    label="Enter your text",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                model_choice = gr.Radio(
                    choices=["Logistic Regression", "Random Forest", "XGBoost"],
                    label="Select Model",
                    value="Logistic Regression"
                )
                predict_btn1 = gr.Button("Analyze", variant="primary")
                
            with gr.Column():
                output1 = gr.Textbox(label="Prediction", lines=2)
                prob_output1 = gr.Label(label="Probability Distribution")
        
        gr.Examples(examples=examples, inputs=text_input1)
        
    with gr.Tab("Ensemble Prediction"):
        with gr.Row():
            with gr.Column():
                text_input2 = gr.Textbox(
                    label="Enter your text",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                predict_btn2 = gr.Button("Analyze with All Models", variant="primary")
                
            with gr.Column():
                output2 = gr.Textbox(label="Predictions", lines=8)
        
        gr.Examples(examples=examples, inputs=text_input2)
        
    with gr.Tab("Detailed Analysis"):
        with gr.Row():
            with gr.Column():
                text_input3 = gr.Textbox(
                    label="Enter your text",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                analyze_btn = gr.Button("Get Detailed Analysis", variant="primary")
                
            with gr.Column():
                detailed_output = gr.Markdown()
        
        gr.Examples(examples=examples, inputs=text_input3)
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## About This Application
        
        This mental health sentiment analysis tool uses three different machine learning models:
        
        1. **Logistic Regression**: A linear model that's fast and interpretable
        2. **Random Forest**: An ensemble of decision trees for better accuracy
        3. **XGBoost**: A powerful gradient boosting algorithm
        
        ### Features:
        - Multi-class classification for various mental health states
        - Ensemble predictions for better reliability
        - Confidence scores for each prediction
        - Support for imbalanced datasets
        
        ### Important Notes:
        - This tool is for educational and research purposes only
        - It should not be used as a substitute for professional mental health diagnosis
        - If you're experiencing mental health issues, please consult a healthcare professional
        
        ### Privacy:
        - No data is stored or transmitted
        - All processing happens locally
        """)
    
    # Connect buttons to functions
    predict_btn1.click(
        single_model_prediction,
        inputs=[text_input1, model_choice],
        outputs=[output1, prob_output1]
    )
    
    predict_btn2.click(
        ensemble_prediction,
        inputs=text_input2,
        outputs=output2
    )
    
    analyze_btn.click(
        analyze_sentiment,
        inputs=text_input3,
        outputs=detailed_output
    )


# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)