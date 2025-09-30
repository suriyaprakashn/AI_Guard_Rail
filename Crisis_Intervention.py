import re
import numpy as np

# Crisis indicators database
CRISIS_INDICATORS = {
    'imminent_self_harm': {
        'keywords': {
            'english': [
                'kill myself', 'end my life', 'suicide now', 'going to jump',
                'taking pills now', 'cutting myself', 'want to die tonight',
                'final goodbye', 'this is the end', 'no way out',
                'ending it all', 'goodbye forever', 'cant take it anymore'
            ]
        },
        'weight': 2.0,
        'urgency': 'IMMEDIATE'
    },
    'self_harm_ideation': {
        'keywords': {
            'english': [
                'suicide', 'want to die', 'end it all', 'no will to live',
                'better off dead', 'harm myself', 'self harm',
                'no reason to live', 'tired of living', 'life is pointless'
            ]
        },
        'weight': 1.5,
        'urgency': 'HIGH'
    },
    'severe_depression': {
        'keywords': {
            'english': [
                'deeply depressed', 'can\'t go on', 'hopeless', 'empty inside',
                'nothing matters', 'crying constantly', 'can\'t get out of bed',
                'overwhelming sadness', 'emotional pain', 'utterly hopeless'
            ]
        },
        'weight': 1.2,
        'urgency': 'HIGH'
    },
    'crisis_support': {
        'keywords': {
            'english': [
                'help me', 'someone save me', 'emergency', 'need help now',
                'anyone there', 'please help', 'I need someone',
                'can\'t handle this', 'breaking down', 'falling apart'
            ]
        },
        'weight': 1.3,
        'urgency': 'HIGH'
    }
}

# Crisis resources by region
CRISIS_RESOURCES = {
    'US': {
        'National Suicide Prevention Lifeline': '988',
        'Crisis Text Line': 'Text HOME to 741741',
        'Emergency Services': '911'
    },
    'UK': {
        'Samaritans': '116 123',
        'Emergency Services': '999'
    },
    'CA': {
        'Canada Suicide Prevention Service': '1-833-456-4566',
        'Emergency Services': '911'
    },
    'international': {
        'International Suicide Hotlines': 'https://findahelpline.com',
        'Befrienders Worldwide': 'https://www.befrienders.org'
    }
}

def assess_crisis_risk(text, abuse_detected=False, escalation_detected=False, conversation_context=None):
    """
    Comprehensive crisis risk assessment
    """
    text_lower = text.lower()
    
    # Keyword-based detection
    keyword_matches = detect_crisis_keywords(text_lower)
    
    # Contextual analysis
    contextual_risk = analyze_contextual_risk(text_lower, conversation_context)
    
    # Behavioral patterns
    behavioral_risk = analyze_behavioral_patterns(text_lower, abuse_detected)
    
    # Sentiment and emotional intensity
    emotional_risk = analyze_emotional_intensity(text_lower)
    
    # Combined risk score
    total_risk_score = (
        keyword_matches['total_score'] * 0.4 +
        contextual_risk * 0.25 +
        behavioral_risk * 0.2 +
        emotional_risk * 0.15
    )
    
    # Adjust for escalation context
    if escalation_detected:
        total_risk_score += 0.2
    
    total_risk_score = min(total_risk_score, 1.0)
    
    # Determine risk level and intervention
    risk_assessment = determine_risk_level(total_risk_score, keyword_matches)
    
    return {
        **risk_assessment,
        'keyword_matches': keyword_matches['matches'],
        'contextual_risk_score': contextual_risk,
        'behavioral_risk_score': behavioral_risk,
        'emotional_risk_score': emotional_risk,
        'total_risk_score': total_risk_score,
        'resources': get_relevant_resources(risk_assessment['risk_level'])
    }

def detect_crisis_keywords(text_lower):
    """Detect crisis keywords"""
    matches = []
    total_score = 0.0
    
    for category, config in CRISIS_INDICATORS.items():
        for language, keywords in config['keywords'].items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Check for negation
                    if not is_negated(text_lower, keyword):
                        matches.append({
                            'category': category,
                            'keyword': keyword,
                            'language': language,
                            'weight': config['weight'],
                            'urgency': config['urgency']
                        })
                        total_score += config['weight']
    
    return {
        'matches': matches,
        'total_score': min(total_score, 10.0)
    }

def is_negated(text, keyword):
    """Simple negation detection"""
    negations = {'not', 'no', 'never', "don't", "won't", "can't", 'no quiero'}
    
    # Find the position of the keyword
    start_pos = text.find(keyword)
    if start_pos == -1:
        return False
    
    # Check previous words for negation
    preceding_text = text[:start_pos].strip()
    words = preceding_text.split()
    
    # Look at last 3 words before the keyword
    for word in words[-3:]:
        if word in negations:
            return True
    
    return False

def analyze_contextual_risk(text, conversation_context):
    """Analyze contextual risk factors"""
    contextual_score = 0.0
    
    # Isolation indicators
    isolation_indicators = [
        'all alone', 'no one cares', 'nobody understands', 'completely alone',
        'no friends', 'family doesn\'t care', 'everyone left me'
    ]
    
    if any(indicator in text for indicator in isolation_indicators):
        contextual_score += 0.4
    
    # Hopelessness indicators
    hopelessness_indicators = [
        'nothing will help', 'no solution', 'no way out', 'pointless',
        'never get better', 'always be this way', 'things will never change'
    ]
    
    if any(indicator in text for indicator in hopelessness_indicators):
        contextual_score += 0.3
    
    # Recent crisis mentions in conversation history
    if conversation_context and 'recent_crisis_mentions' in conversation_context:
        contextual_score += min(conversation_context['recent_crisis_mentions'] * 0.1, 0.3)
    
    return min(contextual_score, 1.0)

def analyze_behavioral_patterns(text, abuse_detected):
    """Analyze behavioral risk patterns"""
    behavioral_score = 0.0
    
    # Self-directed abuse or self-blame
    self_directed_patterns = [
        'i am stupid', 'i\'m worthless', 'hate myself', 'i deserve this',
        'my fault', 'i\'m a failure', 'useless person', 'i ruin everything'
    ]
    
    if any(pattern in text for pattern in self_directed_patterns):
        behavioral_score += 0.6
    
    # Goodbye or final message patterns
    goodbye_patterns = [
        'goodbye', 'farewell', 'last message', 'see you never',
        'taking a long sleep', 'going away forever', 'this is my last'
    ]
    
    if any(pattern in text for pattern in goodbye_patterns):
        behavioral_score += 0.8
    
    # Plan disclosure patterns
    plan_indicators = [
        'going to', 'plan to', 'thinking about', 'decided to',
        'tonight i will', 'this weekend i\'ll', 'tomorrow i\'m going to'
    ]
    
    crisis_keywords_present = any(
        any(keyword in text for keyword in category_config['keywords']['english'])
        for category_config in CRISIS_INDICATORS.values()
        if 'english' in category_config['keywords']
    )
    
    if any(plan_indicator in text for plan_indicator in plan_indicators) and crisis_keywords_present:
        behavioral_score += 0.7
    
    # Self-directed abuse increases risk
    if abuse_detected and any(pattern in text for pattern in self_directed_patterns):
        behavioral_score += 0.3
    
    return min(behavioral_score, 1.0)

def analyze_emotional_intensity(text):
    """Analyze emotional intensity of the text"""
    emotional_score = 0.0
    
    # Intensity indicators
    intensity_indicators = {
        '!!!': 0.3,
        '??': 0.2,
        '...': 0.1,  # Ellipsis can indicate despair
        'crying': 0.4,
        'screaming': 0.5,
        'breaking down': 0.6,
        'cant breathe': 0.4,
        'panic attack': 0.7
    }
    
    for indicator, weight in intensity_indicators.items():
        if indicator in text:
            emotional_score += weight
    
    # All caps emotional outbursts
    if len(text) > 10:
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if caps_ratio > 0.5:
            emotional_score += 0.4
    
    # Repeated punctuation
    if '!!!' in text or '???' in text:
        emotional_score += 0.2
    
    return min(emotional_score, 1.0)

def determine_risk_level(total_score, keyword_matches):
    """Determine appropriate risk level and intervention"""
    has_imminent_keywords = any(
        match['urgency'] == 'IMMEDIATE' for match in keyword_matches['matches']
    )
    
    if total_score >= 2.5 or has_imminent_keywords:
        return {
            'risk_level': 'CRITICAL',
            'intervention_required': True,
            'intervention_type': 'IMMEDIATE_HUMAN',
            'alert_message': 'Critical crisis situation detected - immediate human intervention required',
            'priority': 'HIGHEST'
        }
    elif total_score >= 1.5:
        return {
            'risk_level': 'HIGH',
            'intervention_required': True,
            'intervention_type': 'PRIORITY_HUMAN',
            'alert_message': 'High risk situation - prioritize human review',
            'priority': 'HIGH'
        }
    elif total_score >= 0.8:
        return {
            'risk_level': 'MEDIUM',
            'intervention_required': True,
            'intervention_type': 'ROUTINE_CHECK',
            'alert_message': 'Moderate risk detected - schedule human review',
            'priority': 'MEDIUM'
        }
    else:
        return {
            'risk_level': 'LOW',
            'intervention_required': False,
            'intervention_type': 'MONITOR',
            'alert_message': 'Low risk - continue monitoring',
            'priority': 'LOW'
        }

def get_relevant_resources(risk_level):
    """Get appropriate crisis resources based on risk level"""
    if risk_level in ['MEDIUM', 'HIGH', 'CRITICAL']:
        return [
            "Consider reaching out to mental health professionals",
            "Crisis support is available - you're not alone",
            "National Suicide Prevention Lifeline: 988 (US)",
            "Crisis Text Line: Text HOME to 741741 (US)",
            "International help: https://findahelpline.com"
        ]
    return []