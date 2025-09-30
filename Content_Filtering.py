import re
import textstat

# Age-appropriate content definitions
AGE_PROFILES = {
    'child': {
        'min_age': 5,
        'max_age': 12,
        'restricted_categories': ['violence', 'explicit', 'drugs', 'alcohol', 'weapons', 'bullying', 'graphic_violence'],
        'complexity_threshold': 3.0,
        'allowed_topics': ['education', 'games', 'family', 'school', 'friends']
    },
    'teen': {
        'min_age': 13,
        'max_age': 17,
        'restricted_categories': ['explicit', 'drugs', 'weapons', 'graphic_violence'],
        'complexity_threshold': 6.0,
        'allowed_topics': ['education', 'social', 'relationships', 'hobbies']
    },
    'adult': {
        'min_age': 18,
        'max_age': 100,
        'restricted_categories': [],  # Minimal restrictions for adults
        'complexity_threshold': 10.0,
        'allowed_topics': 'all'
    }
}

# Content category definitions
CONTENT_CATEGORIES = {
    'violence': {
        'keywords': ['fight', 'kill', 'hurt', 'attack', 'violence', 'punch', 'hit', 'beat', 'war', 'battle'],
        'weight': 0.8,
        'severity': 'medium'
    },
    'explicit': {
        'keywords': ['sex', 'naked', 'porn', 'explicit', 'xxx', 'adult content', 'nsfw', 'sexual', 'bedroom'],
        'weight': 0.9,
        'severity': 'high'
    },
    'drugs': {
        'keywords': ['drugs', 'cocaine', 'heroin', 'marijuana', 'weed', 'alcohol', 'drunk', 'beer', 'wine', 'vodka', 'whiskey'],
        'weight': 0.7,
        'severity': 'medium'
    },
    'weapons': {
        'keywords': ['gun', 'knife', 'bomb', 'weapon', 'shoot', 'firearm', 'ammo', 'bullet', 'arsenal'],
        'weight': 0.6,
        'severity': 'high'
    },
    'bullying': {
        'keywords': ['bully', 'tease', 'harass', 'mock', 'make fun of', 'pick on', 'cyberbully'],
        'weight': 0.5,
        'severity': 'medium'
    },
    'graphic_violence': {
        'keywords': ['blood', 'gore', 'mutilate', 'torture', 'brutal', 'horror', 'decapitate', 'massacre'],
        'weight': 0.9,
        'severity': 'high'
    }
}

def filter_content(text, age_profile='teen'):
    """
    Filter content based on age appropriateness
    """
    profile = AGE_PROFILES.get(age_profile, AGE_PROFILES['teen'])
    
    # Check for restricted content
    restricted_found = check_restricted_content(text, profile['restricted_categories'])
    
    # Calculate content complexity
    complexity_score = calculate_content_complexity(text)
    
    # Analyze sentiment and emotional tone
    emotional_tone = analyze_emotional_tone(text)
    
    # Determine appropriateness
    is_appropriate = (
        not restricted_found['has_restricted'] and
        complexity_score <= profile['complexity_threshold'] and
        emotional_tone['appropriate_for_age']
    )
    
    return {
        'is_appropriate': is_appropriate,
        'age_profile': age_profile,
        'restricted_content_found': restricted_found,
        'complexity_score': complexity_score,
        'complexity_appropriate': complexity_score <= profile['complexity_threshold'],
        'emotional_tone': emotional_tone,
        'filter_reason': generate_filter_reason(restricted_found, complexity_score, emotional_tone, profile),
        'suggested_action': 'BLOCK' if not is_appropriate else 'ALLOW'
    }

def check_restricted_content(text, restricted_categories):
    """Check for restricted content categories"""
    text_lower = text.lower()
    found_categories = {}
    
    for category in restricted_categories:
        if category in CONTENT_CATEGORIES:
            category_config = CONTENT_CATEGORIES[category]
            keywords_found = []
            
            for keyword in category_config['keywords']:
                if keyword in text_lower:
                    # Check for educational context (false positive reduction)
                    if not is_educational_context(text_lower, keyword):
                        keywords_found.append(keyword)
            
            if keywords_found:
                found_categories[category] = {
                    'keywords': keywords_found,
                    'severity': category_config['severity'],
                    'count': len(keywords_found),
                    'weight': category_config['weight']
                }
    
    return {
        'has_restricted': len(found_categories) > 0,
        'categories': found_categories,
        'total_restricted_categories': len(found_categories),
        'overall_severity': calculate_overall_severity(found_categories)
    }

def is_educational_context(text, keyword):
    """Check if restricted content appears in educational context"""
    educational_indicators = [
        'education', 'learn about', 'discuss', 'discussion',
        'study', 'research', 'article about', 'news about',
        'academic', 'school', 'university', 'college',
        'documentary', 'educational', 'teaching'
    ]
    
    # If educational context is present, might be a false positive
    return any(indicator in text for indicator in educational_indicators)

def calculate_overall_severity(found_categories):
    """Calculate overall severity of restricted content"""
    if not found_categories:
        return 'none'
    
    severities = [cat['severity'] for cat in found_categories.values()]
    
    if 'high' in severities:
        return 'high'
    elif 'medium' in severities:
        return 'medium'
    else:
        return 'low'

def calculate_content_complexity(text):
    """Calculate content complexity using multiple metrics"""
    if not text.strip():
        return 0.0
    
    # Use textstat for readability metrics
    try:
        flesch_reading = textstat.flesch_reading_ease(text)
        # Convert to 0-10 scale (higher = more complex)
        # Flesch reading ease: 0-30 (very difficult), 90-100 (very easy)
        complexity = max(0, (100 - flesch_reading) / 10)
    except:
        # Fallback calculation
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Complex word ratio (words with > 2 syllables - simplified)
        complex_words = [word for word in words if len(word) > 6]
        complex_ratio = len(complex_words) / len(words)
        
        complexity = (
            avg_sentence_length * 0.4 +
            avg_word_length * 0.3 +
            unique_ratio * 0.3
        ) / 2  # Normalize to roughly 0-10 scale
    
    return min(complexity, 10.0)

def analyze_emotional_tone(text):
    """Analyze emotional tone of content"""
    text_lower = text.lower()
    
    # Positive emotional indicators
    positive_indicators = ['love', 'happy', 'joy', 'fun', 'great', 'awesome', 'wonderful', 'amazing', 'beautiful']
    positive_count = sum(1 for word in positive_indicators if word in text_lower)
    
    # Negative emotional indicators
    negative_indicators = ['hate', 'angry', 'sad', 'terrible', 'awful', 'horrible', 'disgusting', 'upset', 'mad']
    negative_count = sum(1 for word in negative_indicators if word in text_lower)
    
    # Neutral ratio
    total_emotional_words = positive_count + negative_count
    if total_emotional_words > 0:
        negative_ratio = negative_count / total_emotional_words
    else:
        negative_ratio = 0
    
    # Determine if appropriate based on emotional tone
    # For children: allow very little negativity
    # For teens: allow moderate negativity
    # For adults: allow most content
    appropriate_for_age = negative_ratio < 0.7  # Allow some negativity
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'negative_ratio': negative_ratio,
        'appropriate_for_age': appropriate_for_age,
        'emotional_balance': 'positive' if positive_count > negative_count else 'negative'
    }

def generate_filter_reason(restricted_found, complexity_score, emotional_tone, profile):
    """Generate human-readable filter reason"""
    reasons = []
    
    if restricted_found['has_restricted']:
        categories = list(restricted_found['categories'].keys())
        reasons.append(f"Restricted content: {', '.join(categories)}")
    
    if complexity_score > profile['complexity_threshold']:
        reasons.append(f"Content too complex for {profile['min_age']}+ age group")
    
    if not emotional_tone['appropriate_for_age']:
        reasons.append("Emotional tone not appropriate for age group")
    
    return "; ".join(reasons) if reasons else "Content appropriate"

def handle_multilingual_content(text, target_language='english'):
    """
    Basic multilingual content handling
    In production, this would integrate with translation services
    """
    # Simple language detection based on common words
    language_indicators = {
        'english': ['the', 'and', 'you', 'that', 'this', 'with', 'for', 'are'],
        'spanish': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'por'],
        'french': ['le', 'la', 'de', 'et', 'que', 'en', 'un', 'pour']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for lang, indicators in language_indicators.items():
        score = sum(1 for word in indicators if word in text_lower)
        scores[lang] = score
    
    detected_language = max(scores, key=scores.get) if scores else 'english'
    
    return {
        'detected_language': detected_language,
        'confidence': scores[detected_language] / len(language_indicators[detected_language]) if detected_language in language_indicators else 0,
        'needs_translation': detected_language != target_language
    }