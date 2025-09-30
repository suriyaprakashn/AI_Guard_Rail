import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid

# Import our enhanced modules
import data_preprocessing as dp
import Abuse_Language_Detection as ald
import Escalation_Pattern_Recognition as epr
import Crisis_Intervention as ci
import Content_Filtering as cf
import model_evaluation as me

# Configure Streamlit page
st.set_page_config(
    page_title="AI Safety Models POC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'safety_metrics' not in st.session_state:
        st.session_state.safety_metrics = {
            'total_messages': 0,
            'abuse_detected': 0,
            'crisis_flags': 0,
            'content_blocks': 0,
            'escalations_detected': 0
        }
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = str(uuid.uuid4())[:8]
    if 'datasets_loaded' not in st.session_state:
        st.session_state.datasets_loaded = False
    if 'dataset_stats' not in st.session_state:
        st.session_state.dataset_stats = None

init_session_state()

def load_datasets():
    """Load and display dataset information"""
    with st.spinner("Loading cyber-bullying datasets..."):
        combined_df, datasets = dp.load_cyberbullying_datasets()
        if combined_df is not None:
            st.session_state.datasets_loaded = True
            st.session_state.combined_df = combined_df
            st.session_state.datasets = datasets
            st.session_state.dataset_stats = dp.analyze_dataset_statistics(combined_df)
            return True
        else:
            st.error("Could not load datasets")
            return False

def main():
    st.title("🛡️ AI Safety Models - Cyber-Bullying POC")
    st.markdown("""
    This Proof of Concept demonstrates real-time AI safety monitoring trained on cyber-bullying datasets.
    The models are trained on data from Kaggle, Twitter, Wikipedia, and YouTube.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        age_profile = st.selectbox(
            "Age Profile",
            ["child", "teen", "adult"],
            index=1,
            help="Content filtering adjusts based on age appropriateness"
        )
        
        st.header("📊 Dataset Management")
        
        if st.button("Load Cyber-bullying Datasets"):
            if load_datasets():
                st.success("Datasets loaded successfully!")
        
        if st.button("Retrain Models"):
            with st.spinner("Retraining models with latest data..."):
                ald.load_or_train_abuse_model(force_retrain=True)
                st.success("Models retrained successfully")
        
        st.header("🧪 Quick Tests")
        test_messages = [
            "Hello, how are you doing today?",
            "You fuck your dad.",  # From kaggle dataset
            "Call me sexist, but when I go to an auto place, I'd rather talk to a guy",  # From twitter dataset
            "This is not creative. I'm tired of arguing with you.",  # From aggression dataset
            "Does N.e.bodyelse Hear her Crazy ass Screamin",  # From youtube dataset
            "I want to kill myself, everything is hopeless"
        ]
        
        for msg in test_messages:
            if st.button(f"Test: {msg[:30]}...", key=msg):
                st.session_state.test_message = msg
        
        # Display dataset statistics if available
        if st.session_state.datasets_loaded and st.session_state.dataset_stats:
            st.header("📈 Dataset Stats")
            stats = st.session_state.dataset_stats
            st.metric("Total Samples", stats['total_samples'])
            st.metric("Abusive Samples", stats['positive_samples'])
            st.metric("Non-abusive Samples", stats['negative_samples'])
            st.metric("Abuse Rate", f"{(stats['positive_ratio'] * 100):.1f}%")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Real-time Safety Analysis")
        
        # Message input
        message_text = st.text_area(
            "Enter message to analyze:",
            value=st.session_state.get('test_message', ''),
            height=100,
            placeholder="Type your message here...",
            key="message_input"
        )
        
        analyze_button = st.button("Analyze Safety", type="primary")
        
        if analyze_button and message_text.strip():
            with st.spinner("Analyzing message across all safety models..."):
                # Preprocess text
                processed_text = dp.preprocess_text(message_text)
                
                # 1. Abuse Detection
                abuse_result = ald.detect_abuse_text(processed_text)
                
                # 2. Escalation Detection
                escalation_result = epr.analyze_escalation_pattern(
                    st.session_state.current_conversation,
                    processed_text,
                    abuse_result['max_score'],
                    -0.5 if abuse_result['is_abusive'] else 0.1
                )
                
                # 3. Crisis Intervention
                crisis_result = ci.assess_crisis_risk(
                    processed_text,
                    abuse_result['is_abusive'],
                    escalation_result['escalation_detected']
                )
                
                # 4. Content Filtering
                content_result = cf.filter_content(processed_text, age_profile)
                
                # Store results
                analysis_result = {
                    'timestamp': datetime.now().isoformat(),
                    'text': message_text,
                    'processed_text': processed_text,
                    'abuse_detection': abuse_result,
                    'escalation_detection': escalation_result,
                    'crisis_assessment': crisis_result,
                    'content_filtering': content_result,
                    'age_profile': age_profile
                }
                
                # Update conversation history
                if st.session_state.current_conversation not in st.session_state.conversations:
                    st.session_state.conversations[st.session_state.current_conversation] = []
                
                st.session_state.conversations[st.session_state.current_conversation].append(analysis_result)
                
                # Update metrics
                st.session_state.safety_metrics['total_messages'] += 1
                if abuse_result['is_abusive']:
                    st.session_state.safety_metrics['abuse_detected'] += 1
                if crisis_result['intervention_required']:
                    st.session_state.safety_metrics['crisis_flags'] += 1
                if not content_result['is_appropriate']:
                    st.session_state.safety_metrics['content_blocks'] += 1
                if escalation_result['escalation_detected']:
                    st.session_state.safety_metrics['escalations_detected'] += 1
                
                # Display results
                display_analysis_results(analysis_result)
    
    with col2:
        st.header("📈 Safety Dashboard")
        display_safety_metrics()
        
        st.header("🕒 Recent Activity")
        display_recent_activity()
        
        # Evaluation section
        st.header("📋 Model Evaluation")
        me.display_evaluation_results()

def display_analysis_results(analysis):
    """Display comprehensive analysis results"""
    st.subheader("🔍 Safety Analysis Results")
    
    # Overall safety status
    safety_status = determine_overall_safety_status(analysis)
    status_color = {
        "SAFE": "🟢",
        "WARNING": "🟡", 
        "CRITICAL": "🔴",
        "RESTRICTED": "🟠"
    }
    
    st.markdown(f"### {status_color[safety_status]} Overall Safety: **{safety_status}**")
    
    # Model results in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Abuse detection
        st.metric(
            "Abuse Detection", 
            "DETECTED" if analysis['abuse_detection']['is_abusive'] else "CLEAN",
            delta=f"Score: {analysis['abuse_detection']['max_score']:.2f}"
        )
        st.caption(f"Model: {analysis['abuse_detection']['model_used']}")
    
    with col2:
        # Escalation detection
        st.metric(
            "Escalation", 
            "DETECTED" if analysis['escalation_detection']['escalation_detected'] else "NORMAL",
            delta=f"Score: {analysis['escalation_detection']['escalation_score']:.2f}"
        )
    
    with col3:
        # Crisis assessment
        crisis_level = analysis['crisis_assessment']['risk_level']
        st.metric(
            "Crisis Risk", 
            crisis_level,
            delta=f"Score: {analysis['crisis_assessment']['total_risk_score']:.2f}"
        )
    
    with col4:
        # Content filtering
        st.metric(
            "Content Appropriate", 
            "YES" if analysis['content_filtering']['is_appropriate'] else "NO",
            delta=analysis['content_filtering']['age_profile']
        )
    
    # Detailed breakdown
    with st.expander("📊 Detailed Analysis", expanded=True):
        tab1, tab2, tab3, tab4 = st.tabs(["Abuse", "Escalation", "Crisis", "Content"])
        
        with tab1:
            display_abuse_details(analysis['abuse_detection'])
        
        with tab2:
            display_escalation_details(analysis['escalation_detection'])
        
        with tab3:
            display_crisis_details(analysis['crisis_assessment'])
        
        with tab4:
            display_content_details(analysis['content_filtering'])
    
    # Recommended actions
    st.subheader("🎯 Recommended Actions")
    display_recommended_actions(analysis)

def determine_overall_safety_status(analysis):
    """Determine overall safety status based on all models"""
    if analysis['crisis_assessment']['risk_level'] in ['HIGH', 'CRITICAL']:
        return "CRITICAL"
    elif analysis['abuse_detection']['is_abusive']:
        return "WARNING"
    elif not analysis['content_filtering']['is_appropriate']:
        return "RESTRICTED"
    elif analysis['escalation_detection']['escalation_detected']:
        return "WARNING"
    else:
        return "SAFE"

def display_abuse_details(abuse_result):
    """Display abuse detection details"""
    st.write(f"**Model Used**: {abuse_result['model_used']}")
    st.write(f"**Abusive Content Detected**: {abuse_result['is_abusive']}")
    st.write(f"**Confidence Score**: {abuse_result['max_score']:.3f}")
    
    if abuse_result['scores']:
        st.write("**Detailed Scores**:")
        for category, score in abuse_result['scores'].items():
            st.write(f"- {category}: {score:.3f}")
    
    if 'matched_patterns' in abuse_result and abuse_result['matched_patterns']:
        st.write("**Matched Patterns**:")
        for pattern in abuse_result['matched_patterns']:
            st.write(f"- `{pattern}`")

def display_escalation_details(escalation_result):
    """Display escalation detection details"""
    st.write(f"**Escalation Detected**: {escalation_result['escalation_detected']}")
    st.write(f"**Escalation Score**: {escalation_result['escalation_score']:.3f}")
    st.write(f"**Current Intensity**: {escalation_result['current_intensity']:.3f}")
    st.write(f"**Conversation Length**: {escalation_result['conversation_length']} messages")
    
    st.write("**Trend Analysis**:")
    trends = escalation_result['trend_analysis']
    for trend, value in trends.items():
        st.write(f"- {trend}: {value:.3f}")
    
    if escalation_result['patterns_detected']:
        st.write("**Patterns Detected**:")
        for pattern in escalation_result['patterns_detected']:
            st.write(f"- {pattern.replace('_', ' ').title()}")

def display_crisis_details(crisis_result):
    """Display crisis assessment details"""
    st.write(f"**Risk Level**: {crisis_result['risk_level']}")
    st.write(f"**Intervention Required**: {crisis_result['intervention_required']}")
    st.write(f"**Total Risk Score**: {crisis_result['total_risk_score']:.3f}")
    
    st.write("**Component Scores**:")
    st.write(f"- Contextual Risk: {crisis_result['contextual_risk_score']:.3f}")
    st.write(f"- Behavioral Risk: {crisis_result['behavioral_risk_score']:.3f}")
    st.write(f"- Emotional Risk: {crisis_result['emotional_risk_score']:.3f}")
    
    if crisis_result['keyword_matches']:
        st.write("**Keyword Matches**:")
        for match in crisis_result['keyword_matches'][:3]:
            st.write(f"- {match['category']}: '{match['keyword']}' ({match['language']})")

def display_content_details(content_result):
    """Display content filtering details"""
    st.write(f"**Age Profile**: {content_result['age_profile']}")
    st.write(f"**Appropriate Content**: {content_result['is_appropriate']}")
    st.write(f"**Complexity Score**: {content_result['complexity_score']:.2f}")
    st.write(f"**Complexity Appropriate**: {content_result['complexity_appropriate']}")
    
    if not content_result['is_appropriate']:
        st.write(f"**Filter Reason**: {content_result['filter_reason']}")
        st.write(f"**Suggested Action**: {content_result['suggested_action']}")
    
    if content_result['restricted_content_found']['has_restricted']:
        st.write("**Restricted Content Found**:")
        for category, details in content_result['restricted_content_found']['categories'].items():
            st.write(f"- {category}: {details['keywords']}")

def display_recommended_actions(analysis):
    """Display recommended actions based on analysis"""
    actions = []
    
    # Crisis actions (highest priority)
    if analysis['crisis_assessment']['intervention_required']:
        actions.append(f"🚨 {analysis['crisis_assessment']['alert_message']}")
        if analysis['crisis_assessment']['resources']:
            actions.extend([f"📞 {resource}" for resource in analysis['crisis_assessment']['resources']])
    
    # Abuse actions
    if analysis['abuse_detection']['is_abusive']:
        actions.append("⚠️ Flag content for moderator review")
        actions.append("👁️ Monitor user for further abusive behavior")
    
    # Escalation actions
    if analysis['escalation_detection']['escalation_detected']:
        actions.append("📈 Monitor conversation closely for further escalation")
        actions.append("🕒 Consider proactive de-escalation measures")
    
    # Content filtering actions
    if not analysis['content_filtering']['is_appropriate']:
        actions.append(f"🚫 {analysis['content_filtering']['suggested_action']} content")
        actions.append(f"📝 Reason: {analysis['content_filtering']['filter_reason']}")
    
    if not actions:
        actions.append("✅ No safety actions needed - content is safe")
    
    for action in actions:
        st.write(action)

def display_safety_metrics():
    """Display safety metrics dashboard"""
    metrics = st.session_state.safety_metrics
    
    if metrics['total_messages'] == 0:
        st.info("No messages analyzed yet. Send a message to see metrics.")
        return
    
    # Create metrics DataFrame for visualization
    metric_data = {
        'Category': ['Total Messages', 'Abuse Detected', 'Crisis Flags', 'Content Blocks', 'Escalations'],
        'Count': [
            metrics['total_messages'],
            metrics['abuse_detected'],
            metrics['crisis_flags'],
            metrics['content_blocks'],
            metrics['escalations_detected']
        ]
    }
    df = pd.DataFrame(metric_data)
    
    # Bar chart
    fig = px.bar(
        df, 
        x='Category', 
        y='Count',
        title="Safety Metrics Overview",
        color='Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detection rates
    st.subheader("📊 Detection Rates")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        abuse_rate = (metrics['abuse_detected'] / metrics['total_messages']) * 100
        st.metric("Abuse Detection Rate", f"{abuse_rate:.1f}%")
    
    with col2:
        crisis_rate = (metrics['crisis_flags'] / metrics['total_messages']) * 100
        st.metric("Crisis Flag Rate", f"{crisis_rate:.1f}%")
    
    with col3:
        block_rate = (metrics['content_blocks'] / metrics['total_messages']) * 100
        st.metric("Content Block Rate", f"{block_rate:.1f}%")

def display_recent_activity():
    """Display recent conversation activity"""
    if not st.session_state.conversations:
        st.info("No conversation history yet.")
        return
    
    current_conv = st.session_state.conversations.get(st.session_state.current_conversation, [])
    
    if not current_conv:
        st.info("No messages in current conversation.")
        return
    
    # Show last 5 messages
    recent_messages = current_conv[-5:]
    
    for i, msg in enumerate(reversed(recent_messages)):
        with st.expander(f"Message {len(recent_messages)-i}: {msg['text'][:50]}...", expanded=i==0):
            st.write(f"**Time**: {msg['timestamp'][11:19]}")
            st.write(f"**Abuse**: {msg['abuse_detection']['is_abusive']}")
            st.write(f"**Crisis**: {msg['crisis_assessment']['risk_level']}")
            st.write(f"**Content**: {'✅' if msg['content_filtering']['is_appropriate'] else '❌'}")

# Run the application
if __name__ == "__main__":
    main()