import numpy as np
from collections import deque
import re

# Global conversation state
conversation_history = {}
MAX_HISTORY = 10

def analyze_escalation_pattern(conversation_id, current_message, abuse_score, sentiment_score=0):
    """
    Analyze conversation for escalation patterns
    Detects increasing aggression and negativity over time
    """
    # Initialize conversation state if new
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = {
            'messages': deque(maxlen=MAX_HISTORY),
            'abuse_scores': deque(maxlen=MAX_HISTORY),
            'sentiment_scores': deque(maxlen=MAX_HISTORY),
            'intensity_scores': deque(maxlen=MAX_HISTORY),
            'timestamps': deque(maxlen=MAX_HISTORY)
        }
    
    state = conversation_history[conversation_id]
    
    # Calculate current message intensity
    current_intensity = calculate_message_intensity(current_message)
    
    # Update state
    state['messages'].append(current_message)
    state['abuse_scores'].append(abuse_score)
    state['sentiment_scores'].append(sentiment_score)
    state['intensity_scores'].append(current_intensity)
    
    # Analyze patterns
    escalation_score = calculate_escalation_score(state)
    trend_analysis = analyze_trends(state)
    
    return {
        'escalation_detected': escalation_score > 0.7,
        'escalation_score': escalation_score,
        'trend_analysis': trend_analysis,
        'current_intensity': current_intensity,
        'conversation_length': len(state['messages']),
        'patterns_detected': detect_specific_patterns(state)
    }

def calculate_message_intensity(text):
    """
    Calculate intensity of a single message
    """
    intensity = 0.0
    
    # Linguistic intensity indicators
    text_lower = text.lower()
    
    # Exclamation and question marks
    intensity += text.count('!') * 0.2
    intensity += text.count('?') * 0.1
    
    # Capitalization intensity
    if len(text) > 0:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        intensity += caps_ratio * 0.3
    
    # Negative emotion words with weights
    high_intensity_words = ['hate', 'kill', 'destroy', 'furious', 'enrage', 'despise', 'loathe']
    medium_intensity_words = ['angry', 'mad', 'upset', 'annoyed', 'frustrated', 'irritated']
    low_intensity_words = ['dislike', 'bother', 'trouble', 'concern']
    
    for word in high_intensity_words:
        intensity += text_lower.count(word) * 0.3
    
    for word in medium_intensity_words:
        intensity += text_lower.count(word) * 0.15
    
    for word in low_intensity_words:
        intensity += text_lower.count(word) * 0.05
    
    # Message length intensity (very long messages can indicate emotional intensity)
    word_count = len(text.split())
    if word_count > 50:  # Very long message
        intensity += 0.2
    elif word_count > 20:  # Long message
        intensity += 0.1
    
    # Threat indicators
    threat_words = ['kill', 'hurt', 'destroy', 'end', 'finish', 'attack']
    if any(word in text_lower for word in threat_words):
        intensity += 0.3
    
    return min(intensity, 1.0)

def calculate_escalation_score(state):
    """
    Calculate overall escalation score based on conversation history
    """
    if len(state['abuse_scores']) < 3:
        return 0.0
    
    abuse_scores = list(state['abuse_scores'])
    intensity_scores = list(state['intensity_scores'])
    sentiment_scores = list(state['sentiment_scores'])
    
    # Calculate trends
    abuse_trend = calculate_trend(abuse_scores)
    intensity_trend = calculate_trend(intensity_scores)
    sentiment_trend = calculate_trend(sentiment_scores)
    
    # Recent spikes
    recent_abuse_spike = detect_recent_spike(abuse_scores)
    recent_intensity_spike = detect_recent_spike(intensity_scores)
    
    # Volatility (rapid changes)
    abuse_volatility = np.std(abuse_scores) if len(abuse_scores) > 1 else 0
    intensity_volatility = np.std(intensity_scores) if len(intensity_scores) > 1 else 0
    
    # Combined escalation score
    escalation_score = (
        max(0, abuse_trend) * 0.3 +          # Only increasing abuse matters
        max(0, intensity_trend) * 0.25 +     # Only increasing intensity matters
        min(0, sentiment_trend) * 0.2 +      # Only negative sentiment trends matter
        recent_abuse_spike * 0.15 +
        recent_intensity_spike * 0.1
    )
    
    # Adjust for volatility
    volatility_penalty = (abuse_volatility + intensity_volatility) * 0.1
    escalation_score += volatility_penalty
    
    return min(escalation_score, 1.0)

def calculate_trend(values):
    """Calculate linear trend of values"""
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Simple linear regression
    if len(values) == 1:
        return 0.0
    
    slope = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x*x) - np.sum(x)**2)
    return float(slope)

def detect_recent_spike(values, window=3):
    """Detect recent spikes in values"""
    if len(values) < window + 1:
        return 0.0
    
    recent = values[-window:]
    baseline = np.mean(values[:-window]) if len(values) > window else values[0]
    
    if baseline == 0:
        return 0.0
    
    spike_magnitude = (max(recent) - baseline) / (baseline + 1e-8)
    return min(spike_magnitude, 1.0)

def analyze_trends(state):
    """Detailed trend analysis"""
    abuse_scores = list(state['abuse_scores'])
    sentiment_scores = list(state['sentiment_scores'])
    intensity_scores = list(state['intensity_scores'])
    
    return {
        'abuse_trend': calculate_trend(abuse_scores),
        'sentiment_trend': calculate_trend(sentiment_scores),
        'intensity_trend': calculate_trend(intensity_scores),
        'abuse_volatility': np.std(abuse_scores) if abuse_scores else 0,
        'sentiment_volatility': np.std(sentiment_scores) if sentiment_scores else 0,
        'intensity_volatility': np.std(intensity_scores) if intensity_scores else 0
    }

def detect_specific_patterns(state):
    """Detect specific dangerous conversation patterns"""
    patterns = []
    messages = list(state['messages'])
    
    if len(messages) < 2:
        return patterns
    
    # Pattern: Rapid fire messages (potential flooding/overwhelming)
    if len(messages) >= 5:
        # Check if messages are coming in quick succession
        # (This would use timestamps in a real implementation)
        patterns.append('high_frequency_messaging')
    
    # Pattern: Increasing message length
    message_lengths = [len(msg) for msg in messages]
    if calculate_trend(message_lengths) > 0.1:
        patterns.append('increasing_message_length')
    
    # Pattern: Repeated aggressive language
    recent_messages = messages[-3:] if len(messages) >= 3 else messages
    aggressive_count = 0
    for msg in recent_messages:
        if any(word in msg.lower() for word in ['hate', 'stupid', 'idiot', 'kill', 'hurt', 'angry']):
            aggressive_count += 1
    
    if aggressive_count >= 2:
        patterns.append('repeated_aggression')
    
    # Pattern: Escalating threats
    threat_words = ['kill', 'hurt', 'destroy', 'end you', 'attack']
    threat_counts = []
    for msg in messages:
        threat_count = sum(1 for word in threat_words if word in msg.lower())
        threat_counts.append(threat_count)
    
    if calculate_trend(threat_counts) > 0:
        patterns.append('escalating_threats')
    
    # Pattern: Personal attacks becoming more specific
    personal_attack_indicators = ['you are', 'you\'re', 'your family', 'your mother', 'your father']
    personal_attack_counts = []
    for msg in messages:
        attack_count = sum(1 for indicator in personal_attack_indicators if indicator in msg.lower())
        personal_attack_counts.append(attack_count)
    
    if calculate_trend(personal_attack_counts) > 0:
        patterns.append('increasing_personal_attacks')
    
    return patterns

def get_conversation_summary(conversation_id):
    """Get summary of conversation state"""
    if conversation_id not in conversation_history:
        return None
    
    state = conversation_history[conversation_id]
    
    return {
        'message_count': len(state['messages']),
        'recent_abuse_score': state['abuse_scores'][-1] if state['abuse_scores'] else 0,
        'recent_intensity': state['intensity_scores'][-1] if state['intensity_scores'] else 0,
        'average_abuse_score': np.mean(list(state['abuse_scores'])) if state['abuse_scores'] else 0,
        'trend_analysis': analyze_trends(state)
    }

def reset_conversation(conversation_id):
    """Reset conversation history"""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]