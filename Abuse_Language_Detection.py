import torch
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

# Import our enhanced preprocessing
import data_preprocessing as dp

# Global variables for models
abuse_classifier = None
ml_classifier = None
vectorizer = None
model_loaded = False

def load_or_train_abuse_model(force_retrain=False):
    """
    Load pre-trained model or train new one using cyber-bullying datasets
    """
    global abuse_classifier, ml_classifier, vectorizer, model_loaded
    
    model_path = 'models/abuse_detector'
    ml_model_path = 'models/ml_abuse_classifier.joblib'
    vectorizer_path = 'models/vectorizer.joblib'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    if not force_retrain and os.path.exists(ml_model_path):
        try:
            # Load existing models
            ml_classifier = joblib.load(ml_model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Try to load transformer model
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                abuse_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
            else:
                setup_pretrained_model()
                
            model_loaded = True
            print("Loaded existing abuse detection models")
            return
        except Exception as e:
            print(f"Error loading existing models: {e}. Retraining...")
    
    # Train new models using cyber-bullying datasets
    print("Training abuse detection models using cyber-bullying datasets...")
    
    # Load and prepare data
    combined_df, datasets = dp.load_cyberbullying_datasets()
    
    if combined_df is None or len(combined_df) == 0:
        print("No training data available. Using fallback models.")
        setup_fallback_models()
        return
    
    # Train machine learning model
    train_ml_model(combined_df, ml_model_path, vectorizer_path)
    
    # Train transformer model if enough data
    if len(combined_df) >= 500:  # Reduced threshold for smaller datasets
        train_transformer_model(combined_df, model_path)
    else:
        print("Insufficient data for transformer model training. Using pre-trained model.")
        setup_pretrained_model()

def setup_fallback_models():
    """Setup fallback models when no training data is available"""
    global abuse_classifier, ml_classifier, vectorizer, model_loaded
    
    try:
        # Try to load pre-trained model
        abuse_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            return_all_scores=True
        )
        print("Using pre-trained toxic-bert model")
    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        abuse_classifier = None
    
    # Create simple vectorizer and classifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    ml_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    model_loaded = True

def setup_pretrained_model():
    """Setup pre-trained model without fine-tuning"""
    global abuse_classifier, model_loaded
    
    try:
        abuse_classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert",
            return_all_scores=True
        )
        print("Using pre-trained toxic-bert model")
    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        abuse_classifier = None
    
    model_loaded = True

def train_ml_model(combined_df, model_path, vectorizer_path):
    """Train traditional ML model with TF-IDF features"""
    global ml_classifier, vectorizer
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test = dp.prepare_training_data(combined_df, balance_data=True)
        
        if X_train is None:
            print("No training data available for ML model")
            return
        
        # Create TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Extract additional features
        train_features = dp.extract_advanced_features(X_train)
        test_features = dp.extract_advanced_features(X_test)
        
        # Combine TF-IDF with additional features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf, train_features.values])
        X_test_combined = hstack([X_test_tfidf, test_features.values])
        
        # Train Random Forest classifier
        ml_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=20,
            min_samples_split=5
        )
        ml_classifier.fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = ml_classifier.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"ML Model Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        # Save models
        joblib.dump(ml_classifier, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved ML model to {model_path}")
        
    except Exception as e:
        print(f"Error training ML model: {e}")
        import traceback
        traceback.print_exc()
        ml_classifier = None
        vectorizer = None

def train_transformer_model(combined_df, model_path):
    """Fine-tune transformer model on cyber-bullying data"""
    global abuse_classifier
    
    try:
        # Use a smaller, faster model for fine-tuning
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
        
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Prepare data
        X_train, X_test, y_train, y_test = dp.prepare_training_data(combined_df, balance_data=True)
        
        if X_train is None:
            print("No training data available for transformer model")
            return
        
        # Tokenize data - use smaller max_length for efficiency
        train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
        test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)
        
        # Create torch datasets
        import torch
        class CyberbullyingDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = CyberbullyingDataset(train_encodings, y_train)
        test_dataset = CyberbullyingDataset(test_encodings, y_test)
        
        # Training arguments with smaller batch size for memory efficiency
        training_args = TrainingArguments(
            output_dir=model_path,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            load_best_model_at_end=True
        )
        
        # Train model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer
        )
        
        print("Starting transformer model training...")
        trainer.train()
        
        # Save model
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Create pipeline for inference
        abuse_classifier = pipeline("text-classification", model=model_path, tokenizer=tokenizer, return_all_scores=True)
        
        print(f"Fine-tuned transformer model saved to {model_path}")
        
    except Exception as e:
        print(f"Error training transformer model: {e}")
        import traceback
        traceback.print_exc()
        setup_pretrained_model()

def detect_abuse_text(text):
    """
    Detect abusive content using trained models with fallbacks
    """
    if not model_loaded:
        load_or_train_abuse_model()
    
    # Preprocess text
    processed_text = dp.preprocess_text(text)
    
    # Try transformer model first
    if abuse_classifier is not None:
        try:
            results = abuse_classifier(processed_text)[0]
            abuse_scores = {}
            
            for result in results:
                label = result['label']
                score = result['score']
                abuse_scores[label] = score
            
            # Determine if abusive based on thresholds
            toxic_threshold = 0.7
            is_abusive = any(score > toxic_threshold for score in abuse_scores.values())
            max_score = max(abuse_scores.values()) if abuse_scores else 0
            
            return {
                'is_abusive': is_abusive,
                'scores': abuse_scores,
                'max_score': max_score,
                'model_used': 'transformer',
                'confidence': max_score,
                'error': None
            }
        except Exception as e:
            print(f"Transformer model error: {e}")
    
    # Try ML model as fallback
    if ml_classifier is not None and vectorizer is not None:
        try:
            # Create features for ML model
            tfidf_features = vectorizer.transform([processed_text])
            advanced_features = dp.extract_advanced_features([processed_text])
            
            # Combine features
            from scipy.sparse import hstack
            combined_features = hstack([tfidf_features, advanced_features.values])
            
            prediction = ml_classifier.predict(combined_features)[0]
            probability = ml_classifier.predict_proba(combined_features)[0][1]
            
            return {
                'is_abusive': bool(prediction),
                'scores': {'ml_classifier': probability},
                'max_score': probability,
                'model_used': 'ml_model',
                'confidence': probability,
                'error': None
            }
        except Exception as e:
            print(f"ML model error: {e}")
    
    # Final fallback: rule-based detection
    return rule_based_abuse_detection(processed_text)

def rule_based_abuse_detection(text):
    """
    Enhanced rule-based abuse detection optimized for your datasets
    """
    text_lower = text.lower()
    
    # Comprehensive abusive patterns from your datasets
    abusive_patterns = [
        # Direct insults and profanity from kaggle dataset
        r'\b(fuck|shit|asshole|bastard|bitch|whore|slut|dick|pussy|cunt)\b',
        r'\b(idiot|stupid|moron|retard|dumbass|fool)\b',
        
        # Threats and violent language
        r'\b(kill|hurt|beat|fight|punch|destroy|harm)\b.*\b(you|u|yourself)\b',
        
        # Appearance-based bullying
        r'\b(ugly|fat|skinny|disgusting|gross|hideous)\b',
        
        # Ability-based bullying  
        r'\b(useless|worthless|failure|loser|incompetent)\b',
        
        # Identity-based attacks (from twitter racism/sexism)
        r'\b(sexist|racist|nigger|fag|faggot|chink|spic|kike)\b',
        
        # Social exclusion patterns
        r'\b(nobody likes you|everyone hates you|you have no friends)\b',
        
        # Cyber-bullying specific patterns
        r'\b(kill yourself|kys|end your life|you should die)\b'
    ]
    
    # Platform-specific patterns from your datasets
    platform_patterns = [
        # Twitter patterns
        r'\b(rt\s+@\w+\s+.*sexist|racist)\b',
        r'call me sexist',
        r'wrong,\s+isis',
        
        # Wikipedia discussion patterns
        r'\b(this is not creative|tired of arguing|double standard)\b',
        
        # YouTube patterns  
        r'\b(crazy ass|stupid hoe|motherfucka)\b'
    ]
    
    all_patterns = abusive_patterns + platform_patterns
    matches = []
    
    for pattern in all_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matches.append(pattern)
    
    is_abusive = len(matches) > 0
    confidence = min(len(matches) * 0.2, 1.0)
    
    # Adjust confidence based on context
    confidence_adj, is_educational, is_self_reflective = handle_ambiguous_cases(text_lower)
    confidence *= confidence_adj
    
    return {
        'is_abusive': is_abusive,
        'scores': {'rule_based': confidence},
        'max_score': confidence,
        'model_used': 'rule_based',
        'matched_patterns': matches[:3],  # Limit to first 3 matches
        'context_adjustment': confidence_adj,
        'error': None
    }

def handle_ambiguous_cases(text):
    """Handle ambiguous language that might be false positives"""
    # Educational or discussion contexts (common in Wikipedia datasets)
    educational_contexts = [
        'discuss', 'discussion', 'education', 'learn', 'study',
        'clinical', 'therapy', 'psychology', 'research', 'article',
        'news', 'report', 'analysis', 'wikipedia', 'edit',
        'neutral', 'point of view', 'citation needed'
    ]
    
    # Self-reflection patterns
    self_reflection = [
        'i feel', 'i am', 'i think', 'in my opinion',
        'personally', 'from my perspective', 'i believe'
    ]
    
    # Quoted speech or examples
    quote_indicators = [
        'said', 'told', 'according to', 'example', 'for instance',
        'quote', 'mentioned', 'stated', 'as per'
    ]
    
    is_educational = any(context in text for context in educational_contexts)
    is_self_reflective = any(pattern in text for pattern in self_reflection)
    is_quoted = any(indicator in text for indicator in quote_indicators)
    
    # Reduce confidence for non-abusive contexts
    if is_educational or is_self_reflective or is_quoted:
        confidence_adjustment = 0.3
    else:
        confidence_adjustment = 1.0
    
    return confidence_adjustment, is_educational, is_self_reflective

def get_dataset_statistics():
    """Get statistics about the loaded datasets"""
    combined_df, datasets = dp.load_cyberbullying_datasets()
    if combined_df is None:
        return "No datasets available"
    
    return dp.analyze_dataset_statistics(combined_df)

# Initialize models when module is imported
load_or_train_abuse_model()