# 🏗️ System Architecture

## Overview
This document describes the architecture and data flow of the AI Safety Models POC.

## Data Flow Diagrams

### 1. Main System Flow

```mermaid
graph TD
    A[User Input] --> B[Preprocessing]
    B --> C[Safety Models]
    C --> D[Integration]
    D --> E[Results Display]
    
    C --> C1[Abuse Detection]
    C --> C2[Escalation Detection]
    C --> C3[Crisis Intervention]
    C --> C4[Content Filtering]
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style C1 fill:#ffebee
    style C2 fill:#e8f5e8
    style C3 fill:#fff3e0
    style C4 fill:#f3e5f5
```
Complete Mermaid Diagrams for All Flows:
1. Main System Data Flow

```mermaid
flowchart TD
    A[User Input<br/>Raw Message, Age Profile] --> B[Text Preprocessing]
    B --> C[Multi-Model Analysis]
    
    C --> D1[Abuse Detection]
    C --> D2[Escalation Detection]
    C --> D3[Crisis Intervention]
    C --> D4[Content Filtering]
    
    D1 --> E[Results Integration]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F[Safety Status & Actions]
    F --> G[Real-time Dashboard]
    
    style A fill:#bbdefb
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style F fill:#ffcdd2
    style G fill:#d1c4e9
```
2. Preprocessing Pipeline

```mermaid
flowchart LR
    A[Raw Text Input] --> B[Lowercase Conversion]
    B --> C[URL Removal]
    C --> D[Mention Handling]
    D --> E[Emoji Translation]
    E --> F[Slang Expansion]
    F --> G[Whitespace Cleaning]
    G --> H[Clean Text Output]
    
    style A fill:#ffebee
    style H fill:#e8f5e8
```
3. Multi-Layered Abuse Detection

```mermaid
flowchart TD
    A[Processed Text] --> B{Transformer Model}
    B -->|Success| C[Confidence Score > 0.7]
    B -->|Failure| D{ML Model Fallback}
    
    C --> F[Final Result]
    D -->|Success| E[ML Prediction]
    D -->|Failure| G[Rule-based Detection]
    
    E --> F
    G --> F
    
    F --> H[Abuse Decision]
    
    style B fill:#fff3e0
    style D fill:#e8f5e8
    style G fill:#ffebee
    style H fill:#bbdefb
```
4. Conversation Escalation Tracking

```mermaid
flowchart TD
    A[Current Message] --> B[Intensity Calculation]
    B --> C[Update Conversation State]
    
    C --> D[Trend Analysis]
    D --> E[Pattern Detection]
    
    E --> F[Calculate Escalation Score]
    F --> G{Escalation > 0.7?}
    G -->|Yes| H[🚨 Escalation Detected]
    G -->|No| I[✅ Normal Conversation]
    
    C --> J[Maintain 10-message Window]
    J --> C
    
    style H fill:#ffcdd2
    style I fill:#c8e6c9
```
5. Crisis Intervention Flow

```mermaid
flowchart TD
    A[Message + Context] --> B[Keyword Matching]
    A --> C[Behavioral Analysis]
    A --> D[Contextual Analysis]
    A --> E[Emotional Intensity]
    
    B --> F[Combine Risk Scores]
    C --> F
    D --> F
    E --> F
    
    F --> G{Risk Assessment}
    
    G -->|Score ≥ 2.5| H[🔴 CRITICAL]
    G -->|Score ≥ 1.5| I[🟠 HIGH]
    G -->|Score ≥ 0.8| J[🟡 MEDIUM]
    G -->|Score < 0.8| K[🟢 LOW]
    
    H --> L[🚨 Immediate Intervention]
    I --> M[⚠️ Priority Review]
    
    style H fill:#ff1744
    style I fill:#ff9100
    style J fill:#ffea00
    style K fill:#00e676
```
