import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_abuse_detection(y_true, y_pred, y_scores=None, model_name="Abuse Detection"):
    """Comprehensive evaluation for abuse detection models"""
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Class-specific metrics
    metrics['precision_positive'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_positive'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_positive'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Detailed classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Additional metrics if probability scores are available
    if y_scores is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            metrics['auc_pr'] = average_precision_score(y_true, y_scores)
        except Exception as e:
            print(f"Could not calculate AUC scores: {e}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Abusive', 'Abusive'],
                yticklabels=['Not Abusive', 'Abusive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def generate_comprehensive_report(metrics_dict, model_name):
    """Generate comprehensive evaluation report"""
    report = f"# Comprehensive Evaluation Report: {model_name}\n\n"
    
    # Summary metrics
    report += "## 📊 Summary Metrics\n"
    summary_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'precision_positive', 'recall_positive', 'f1_positive']
    
    for metric in summary_metrics:
        if metric in metrics_dict:
            report += f"- **{metric.replace('_', ' ').title()}**: {metrics_dict[metric]:.4f}\n"
    
    # AUC scores if available
    if 'auc_roc' in metrics_dict:
        report += f"- **AUC ROC**: {metrics_dict['auc_roc']:.4f}\n"
    if 'auc_pr' in metrics_dict:
        report += f"- **AUC PR**: {metrics_dict['auc_pr']:.4f}\n"
    
    # Detailed classification report
    if 'classification_report' in metrics_dict:
        report += "\n## 📋 Detailed Classification Report\n"
        cr = metrics_dict['classification_report']
        
        for label, scores in cr.items():
            if isinstance(scores, dict) and label in ['0', '1']:
                label_name = "Not Abusive" if label == '0' else "Abusive"
                report += f"\n### **{label_name}**\n"
                for score_name, score_value in scores.items():
                    if score_name in ['precision', 'recall', 'f1-score', 'support']:
                        report += f"- **{score_name}**: {score_value:.4f}\n"
    
    # Confusion matrix
    if 'confusion_matrix' in metrics_dict:
        report += "\n## 🎯 Confusion Matrix\n"
        cm = metrics_dict['confusion_matrix']
        report += f"""
        ```
        Actual\\Predicted | Not Abusive | Abusive
        -----------------|-------------|---------
        Not Abusive      | {cm[0][0]:^11} | {cm[0][1]:^7}
        Abusive          | {cm[1][0]:^11} | {cm[1][1]:^7}
        ```
        """
    
    return report

def evaluate_on_cyberbullying_data():
    """Evaluate models using the cyber-bullying datasets"""
    import data_preprocessing as dp
    import Abuse_Language_Detection as ald
    
    # Load datasets
    combined_df, datasets = dp.load_cyberbullying_datasets()
    
    if combined_df is None:
        return "No datasets available for evaluation"
    
    # Use a subset for evaluation to avoid memory issues
    sample_size = min(500, len(combined_df))
    sample_df = combined_df.sample(sample_size, random_state=42)
    
    predictions = []
    probabilities = []
    true_labels = sample_df['label'].values
    
    print(f"Evaluating on {len(sample_df)} samples...")
    
    for i, text in enumerate(sample_df['text']):
        if i % 50 == 0:
            print(f"Processed {i}/{len(sample_df)} samples")
        
        result = ald.detect_abuse_text(text)
        predictions.append(1 if result['is_abusive'] else 0)
        probabilities.append(result['max_score'])
    
    # Calculate metrics
    metrics = evaluate_abuse_detection(true_labels, predictions, probabilities)
    
    return generate_comprehensive_report(metrics, "Cyber-bullying Dataset Evaluation")

def evaluate_individual_datasets():
    """Evaluate performance on individual datasets"""
    import data_preprocessing as dp
    import Abuse_Language_Detection as ald
    
    combined_df, datasets = dp.load_cyberbullying_datasets()
    
    if not datasets:
        return "No datasets available for evaluation"
    
    reports = []
    
    for dataset_name, df in datasets.items():
        if len(df) < 10:  # Skip very small datasets
            continue
            
        sample_size = min(100, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        
        predictions = []
        true_labels = sample_df['label'].values
        
        for text in sample_df['text']:
            result = ald.detect_abuse_text(text)
            predictions.append(1 if result['is_abusive'] else 0)
        
        metrics = evaluate_abuse_detection(true_labels, predictions)
        report = generate_comprehensive_report(metrics, f"{dataset_name} Dataset")
        reports.append(report)
    
    return "\n\n".join(reports)

# Example usage in Streamlit app
def display_evaluation_results():
    """Display evaluation results in Streamlit"""
    import streamlit as st
    
    st.header("Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Evaluation on Combined Data"):
            with st.spinner("Evaluating models on combined datasets..."):
                report = evaluate_on_cyberbullying_data()
                st.markdown(report)
    
    with col2:
        if st.button("Evaluate Individual Datasets"):
            with st.spinner("Evaluating models on individual datasets..."):
                report = evaluate_individual_datasets()
                st.markdown(report)