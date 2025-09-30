import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os

def load_cyberbullying_datasets(data_path='data/'):
    """
    Load and combine all cyber-bullying datasets with exact column names
    """
    datasets = {}
    
    # Define file names and their specific handling
    dataset_files = {
        'kaggle': 'kaggle_parsed_dataset.csv',
        'aggression': 'aggression_parsed_dataset.csv',
        'attack': 'attack_parsed_dataset.csv',
        'toxicity': 'toxicity_parsed_dataset.csv', 
        'twitter': 'twitter_parsed_dataset.csv',
        'youtube': 'youtube_parsed_dataset.csv'
    }
    
    combined_data = []
    
    for dataset_name, filename in dataset_files.items():
        file_path = os.path.join(data_path, filename)
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded {dataset_name}: {len(df)} rows, columns: {list(df.columns)}")
                
                # Handle each dataset's specific format
                df_processed = process_specific_dataset(df, dataset_name)
                if df_processed is not None:
                    combined_data.append(df_processed)
                    datasets[dataset_name] = df_processed
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    
    # Combine all datasets
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"Total combined dataset: {len(combined_df)} rows")
        print(f"Label distribution: {combined_df['label'].value_counts()}")
        return combined_df, datasets
    else:
        print("No datasets were successfully loaded")
        return None, {}

def process_specific_dataset(df, dataset_name):
    """
    Process each dataset according to its specific format
    """
    df_processed = df.copy()
    
    # Handle each dataset's specific column structure
    if dataset_name == 'kaggle':
        # kaggle: index, oh_label, Date, Text
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    elif dataset_name == 'aggression':
        # aggression: index, Text, ed_label_0, ed_label_1, oh_label
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    elif dataset_name == 'attack':
        # attack: index, Text, ed_label_0, ed_label_1, oh_label  
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    elif dataset_name == 'toxicity':
        # toxicity: index, Text, ed_label_0, ed_label_1, oh_label
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    elif dataset_name == 'twitter':
        # twitter: index, id, Text, Annotation, oh_label
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    elif dataset_name == 'youtube':
        # youtube: index, UserIndex, Text, Number of Comments, Number of Subscribers, Membership Duration, Number of Uploads, Profanity in UserID, Age, oh_label
        if 'Text' in df_processed.columns and 'oh_label' in df_processed.columns:
            df_processed = df_processed[['Text', 'oh_label']].rename(columns={'Text': 'text', 'oh_label': 'label'})
    
    else:
        print(f"Unknown dataset format: {dataset_name}")
        return None
    
    # Clean and validate the processed dataset
    df_processed = clean_dataset(df_processed)
    return df_processed

def clean_dataset(df):
    """Clean and validate the dataset"""
    if df is None or len(df) == 0:
        return None
    
    # Remove rows with missing text
    df = df.dropna(subset=['text'])
    
    # Convert text to string
    df['text'] = df['text'].astype(str)
    
    # Remove empty texts
    df = df[df['text'].str.strip() != '']
    
    # Ensure label is integer (0 or 1)
    if df['label'].dtype != 'int64':
        # Convert to binary: any positive value is 1, else 0
        df['label'] = (pd.to_numeric(df['label'], errors='coerce') > 0).astype(int)
    
    # Remove any rows where label conversion failed
    df = df.dropna(subset=['label'])
    
    return df

def preprocess_text(text):
    """
    Enhanced text preprocessing for cyber-bullying detection
    Handles the specific patterns in your datasets
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions but keep content
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbols but keep the words
    text = re.sub(r'#', '', text)
    
    # Handle markdown-style backticks (common in Wikipedia datasets)
    text = re.sub(r'`{2,3}', ' ', text)  # Remove ``` and ``
    text = re.sub(r'`', "'", text)  # Replace single backticks with quotes
    
    # Handle quotes and special characters
    text = re.sub(r'\"\"', '"', text)
    text = re.sub(r"\'\'", "'", text)
    
    # Handle RT (retweet) patterns from Twitter
    text = re.sub(r'^rt\s+', '', text)
    
    # Handle emojis - convert to text descriptions
    text = handle_emojis(text)
    
    # Handle common internet slang and abbreviations
    text = expand_slang(text)
    
    # Remove extra whitespace and clean up
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def handle_emojis(text):
    """Convert emojis to their text descriptions"""
    emoji_dict = {
        '😂': ' laughing_crying ',
        '😭': ' crying ',
        '😡': ' angry ',
        '🤬': ' swearing ',
        '❤️': ' love ',
        '💔': ' broken_heart ',
        '😊': ' happy ',
        '😢': ' sad ',
        '😠': ' mad ',
        '👊': ' punch ',
        '🔪': ' knife ',
        '💀': ' skull ',
        '🖕': ' middle_finger '
    }
    
    for emoji, description in emoji_dict.items():
        text = text.replace(emoji, description)
    
    return text

def expand_slang(text):
    """Expand common internet slang and abbreviations"""
    slang_dict = {
        'u': ' you ',
        'ur': ' your ',
        'r': ' are ',
        'btw': ' by_the_way ',
        'lol': ' laughing_out_loud ',
        'omg': ' oh_my_god ',
        'wtf': ' what_the_fuck ',
        'idk': ' i_dont_know ',
        'smh': ' shaking_my_head ',
        'tbh': ' to_be_honest ',
        'af': ' as_fuck ',
        'fr': ' for_real ',
        'gtfo': ' get_the_fuck_out ',
        'stfu': ' shut_the_fuck_up ',
        'tf': ' the_fuck ',
        'fml': ' fuck_my_life ',
        'irl': ' in_real_life ',
        'jk': ' just_kidding ',
        'nvm': ' never_mind ',
        'ofc': ' of_course ',
        'ppl': ' people ',
        'thx': ' thanks ',
        'yolo': ' you_only_live_once ',
        'imo': ' in_my_opinion ',
        'brb': ' be_right_back ',
        'np': ' no_problem '
    }
    
    words = text.split()
    processed_words = []
    
    for word in words:
        # Remove repeated characters (e.g., "looooool" -> "lool")
        word = re.sub(r'(.)\1{2,}', r'\1\1', word)
        
        # Handle word boundaries for exact matches
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in slang_dict:
            processed_words.append(slang_dict[clean_word])
        else:
            processed_words.append(word)
    
    return ' '.join(processed_words)

def prepare_training_data(combined_df, test_size=0.2, balance_data=True):
    """
    Prepare data for model training with optional balancing
    """
    if combined_df is None or len(combined_df) == 0:
        return None, None, None, None
    
    # Remove empty texts
    combined_df = combined_df[combined_df['text'].str.strip() != '']
    
    print(f"Dataset before balancing: {len(combined_df)} rows")
    print(f"Label distribution: {combined_df['label'].value_counts()}")
    
    if balance_data:
        # Balance the dataset
        positive_samples = combined_df[combined_df['label'] == 1]
        negative_samples = combined_df[combined_df['label'] == 0]
        
        print(f"Positive samples: {len(positive_samples)}, Negative samples: {len(negative_samples)}")
        
        # Undersample majority class
        min_samples = min(len(positive_samples), len(negative_samples))
        if min_samples > 0:
            positive_balanced = positive_samples.sample(min_samples, random_state=42)
            negative_balanced = negative_samples.sample(min_samples, random_state=42)
            balanced_df = pd.concat([positive_balanced, negative_balanced])
        else:
            balanced_df = combined_df
        
        print(f"Balanced dataset: {len(balanced_df)} rows")
    else:
        balanced_df = combined_df
    
    # Preprocess all texts
    balanced_df['processed_text'] = balanced_df['text'].apply(preprocess_text)
    
    # Split features and labels
    X = balanced_df['processed_text'].values
    y = balanced_df['label'].values
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def extract_advanced_features(texts):
    """
    Extract advanced features for machine learning models
    """
    features = []
    
    for text in texts:
        feature_dict = {}
        
        # Basic text features
        feature_dict['text_length'] = len(text)
        feature_dict['word_count'] = len(text.split())
        
        # Sentiment indicators
        feature_dict['exclamation_count'] = text.count('!')
        feature_dict['question_count'] = text.count('?')
        feature_dict['all_caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Cyber-bullying specific features
        feature_dict['offensive_word_count'] = count_offensive_words(text)
        feature_dict['threat_indicator'] = detect_threat_language(text)
        feature_dict['insult_indicator'] = detect_insult_language(text)
        feature_dict['profanity_count'] = count_profanity(text)
        
        # Conversation features
        feature_dict['has_mention'] = 1 if '@' in text else 0
        feature_dict['has_hashtag'] = 1 if '#' in text else 0
        feature_dict['has_url'] = 1 if 'http' in text else 0
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def count_offensive_words(text):
    """Count offensive words in text"""
    offensive_words = [
        'idiot', 'stupid', 'moron', 'retard', 'fuck', 'shit', 'ass', 'bitch', 
        'whore', 'slut', 'dick', 'pussy', 'bastard', 'damn', 'hell', 'crap',
        'fucking', 'shitty', 'asshole', 'dumbass', 'motherfucker'
    ]
    
    return sum(1 for word in offensive_words if word in text.lower())

def count_profanity(text):
    """Count profanity words"""
    profanity_words = [
        'fuck', 'shit', 'ass', 'bitch', 'whore', 'slut', 'dick', 'pussy',
        'bastard', 'cunt', 'damn', 'hell'
    ]
    
    return sum(1 for word in profanity_words if word in text.lower())

def detect_threat_language(text):
    """Detect threatening language"""
    threat_indicators = ['kill', 'hurt', 'beat', 'fight', 'punch', 'destroy', 'end you', 'harm']
    return 1 if any(indicator in text.lower() for indicator in threat_indicators) else 0

def detect_insult_language(text):
    """Detect insulting language"""
    insult_indicators = ['ugly', 'fat', 'dumb', 'worthless', 'useless', 'loser', 'failure', 'stupid']
    return 1 if any(indicator in text.lower() for indicator in insult_indicators) else 0

def analyze_dataset_statistics(combined_df):
    """Analyze and display dataset statistics"""
    if combined_df is None:
        return "No data available"
    
    stats = {
        'total_samples': len(combined_df),
        'positive_samples': combined_df['label'].sum(),
        'negative_samples': len(combined_df) - combined_df['label'].sum(),
        'positive_ratio': combined_df['label'].mean(),
        'avg_text_length': combined_df['text'].str.len().mean(),
        'avg_word_count': combined_df['text'].str.split().str.len().mean()
    }
    
    return stats