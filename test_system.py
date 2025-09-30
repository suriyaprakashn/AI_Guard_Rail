#!/usr/bin/env python3
"""
Quick test script to verify all modules work together
"""

import Abuse_Language_Detection as ald
import Escalation_Pattern_Recognition as epr
import Crisis_Intervention as ci
import Content_Filtering as cf
import data_preprocessing as dp

def test_all_models():
    """Test all safety models with sample messages"""
    
    test_messages = [
        "Hello, how are you doing today?",
        "You're such an idiot! I hate you!",
        "I'm feeling really depressed and want to end it all",
        "This movie contains explicit adult content and violence",
        "I'm getting really angry about this situation!"
    ]
    
    print("🧪 Testing AI Safety Models...\n")
    
    for i, message in enumerate(test_messages):
        print(f"--- Test {i+1}: '{message}' ---")
        
        # Preprocess
        processed = dp.preprocess_text(message)
        print(f"Processed: {processed}")
        
        # Abuse detection
        abuse_result = ald.detect_abuse_text(message)
        print(f"Abuse: {abuse_result['is_abusive']} (score: {abuse_result['max_score']:.3f})")
        
        # Escalation detection
        escalation_result = epr.analyze_escalation_pattern(
            "test_conversation", 
            message, 
            abuse_result['max_score']
        )
        print(f"Escalation: {escalation_result['escalation_detected']} (score: {escalation_result['escalation_score']:.3f})")
        
        # Crisis intervention
        crisis_result = ci.assess_crisis_risk(
            message,
            abuse_result['is_abusive'],
            escalation_result['escalation_detected']
        )
        print(f"Crisis: {crisis_result['risk_level']} (intervention: {crisis_result['intervention_required']})")
        
        # Content filtering
        content_result = cf.filter_content(message, 'teen')
        print(f"Content: {'Appropriate' if content_result['is_appropriate'] else 'Blocked'}")
        
        print()

if __name__ == "__main__":
    test_all_models()