# Neural Collaborative Filtering (NCF) Integration
## Smart Nutrition Analyzer - Enhanced with AI-Powered Recommendations

### Overview
This module integrates Neural Collaborative Filtering (NCF) into the Smart Health Dashboard to provide personalized food recommendations based on user health profiles and disease predictions.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID RECOMMENDATION SYSTEM                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    USER HEALTH DATA INPUT                        │
│  Age, Gender, BMI, HbA1c, Glucose, Sodium, Potassium, BP,     │
│  Serum Creatinine, Activity Level, Dietary Preferences          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              DISEASE PREDICTION (Random Forest)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Diabetes   │  │   Kidney    │  │   Obesity   │             │
│  │  Predictor  │  │  Predictor  │  │  Predictor  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              NUTRITION FILTERING MODULE                         │
│  Disease-Specific Food Restrictions:                           │
│  - Diabetes: Low sugar, controlled carbs                       │
│  - Kidney Disease: Low sodium, potassium, protein               │
│  - Obesity: Low calorie, low fat, high fiber                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         NEURAL COLLABORATIVE FILTERING (NCF)                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │  User Embedding Layer (32-dim)                 │            │
│  │  Item Embedding Layer (32-dim)                 │            │
│  └─────────────────────────────────────────────────┘            │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Concatenation Layer                            │            │
│  │  Hidden Layers: [64, 32, 16]                    │            │
│  │  Activation: ReLU                                │            │
│  │  Dropout: 0.2                                   │            │
│  └─────────────────────────────────────────────────┘            │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Output Layer (Rating Prediction)               │            │
│  │  Loss: MSE                                     │            │
│  │  Optimizer: Adam (lr=0.001)                    │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TOP-N PERSONALIZED RECOMMENDATIONS                   │
│  Food items ranked by:                                         │
│  1. Predicted rating (NCF)                                     │
│  2. Nutritional suitability                                    │
│  3. Disease compatibility                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
User Input → Disease Prediction → Disease Detection → 
Nutrition Filter → Allowed Foods → NCF Model → 
Personalized Rankings → Top-N Selection → Display
```

### Project Structure

```
ncf_integration/
├── models/
│   ├── ncf_model.py              # NCF model architecture
│   └── train_ncf.py              # Training module
├── data/
│   ├── user_food_interactions.csv  # User-item interaction data
│   ├── food_items.csv             # Food nutritional data
│   └── user_profiles.csv          # User demographic data
├── utils/
│   ├── generate_dataset.py       # Synthetic data generator
│   ├── nutrition_filter.py       # Disease-specific filtering
│   └── hybrid_recommender.py     # Hybrid recommendation pipeline
└── evaluation/
    └── metrics.py                # Evaluation metrics
```

### File Descriptions

#### models/ncf_model.py
- **Purpose**: Implements Neural Collaborative Filtering architecture
- **Key Components**:
  - `NCFModel`: Main NCF class with embedding layers
  - `AdvancedNCF`: Extended version with attention mechanism
  - `build_simple_ncf()`: Simplified architecture for faster training
  - `predict()`: Rating prediction for user-item pairs
  - `recommend_top_n()`: Top-N recommendation generation

#### models/train_ncf.py
- **Purpose**: Handles model training and validation
- **Key Components**:
  - `NCFTrainer`: Training orchestration class
  - `load_data()`: Data loading and preprocessing
  - `train()`: Model training with callbacks
  - `evaluate()`: Model evaluation on test data
  - `plot_training_history()`: Visualization of training metrics

#### utils/generate_dataset.py
- **Purpose**: Generates synthetic user-food interaction dataset
- **Key Components**:
  - `UserFoodDatasetGenerator`: Dataset generation class
  - `generate_user_profiles()`: Creates user demographic profiles
  - `generate_interactions()`: Generates realistic ratings
  - `save_datasets()`: Exports datasets to CSV

#### utils/nutrition_filter.py
- **Purpose**: Filters foods based on disease-specific restrictions
- **Key Components**:
  - `NutritionFilter`: Main filtering class
  - `filter_by_disease()`: Disease-specific food filtering
  - `filter_by_multiple_diseases()`: Multi-disease filtering
  - `score_food_suitability()`: Food suitability scoring

#### utils/hybrid_recommender.py
- **Purpose**: Orchestrates the complete recommendation pipeline
- **Key Components**:
  - `HybridRecommender`: Main recommendation class
  - `predict_diseases()`: Disease prediction using Random Forest
  - `filter_foods_by_diseases()`: Nutrition filtering
  - `get_personalized_recommendations()`: NCF-based recommendations
  - `recommend()`: Complete pipeline execution

#### evaluation/metrics.py
- **Purpose**: Implements recommendation system evaluation metrics
- **Key Components**:
  - `RecommendationMetrics`: Metrics calculation class
  - `precision_at_k()`: Precision@K calculation
  - `recall_at_k()`: Recall@K calculation
  - `ndcg_at_k()`: Normalized Discounted Cumulative Gain
  - `hit_ratio_at_k()`: Hit Ratio@K calculation
  - `rmse()`: Root Mean Square Error
  - `mae()`: Mean Absolute Error

### Training Workflow

1. **Data Generation**
   ```bash
   python ncf_integration/utils/generate_dataset.py
   ```

2. **Model Training**
   ```bash
   python ncf_integration/models/train_ncf.py
   ```

3. **Model Evaluation**
   ```bash
   python ncf_integration/evaluation/metrics.py
   ```

### Testing Workflow

1. **Load Trained Model**
   ```python
   from ncf_integration.models.ncf_model import NCFModel
   ncf = NCFModel(num_users=1000, num_items=500)
   ncf.load_model('ncf_integration/models/ncf_model.keras')
   ```

2. **Generate Recommendations**
   ```python
   from ncf_integration.utils.hybrid_recommender import HybridRecommender
   recommender = HybridRecommender(ncf_model_path='ncf_integration/models/ncf_model.keras')
   result = recommender.recommend(user_id=0, age=45, gender='Male', bmi=28.5, ...)
   ```

### Integration with Streamlit App

The NCF module is integrated into the main Streamlit application (`app.py`):

1. **Import with Fallback**
   ```python
   try:
       from ncf_integration.utils.hybrid_recommender import HybridRecommender
       NCF_AVAILABLE = True
   except ImportError:
       NCF_AVAILABLE = False
   ```

2. **User Toggle**
   - Added checkbox in sidebar: "Use Neural Collaborative Filtering"
   - Allows users to switch between rule-based and AI-powered recommendations

3. **Recommendation Pipeline**
   - Disease prediction remains unchanged (Random Forest)
   - Nutrition filtering applied based on detected diseases
   - NCF provides personalized rankings on filtered foods
   - Fallback to rule-based if NCF fails

### Mathematical Model

#### NCF Architecture

The Neural Collaborative Filtering model combines user and item embeddings through a neural network:

**Embedding Layers:**
```
p_u = E_u[u]  # User embedding (32-dimensional)
q_i = E_i[i]  # Item embedding (32-dimensional)
```

**Concatenation:**
```
h = [p_u; q_i]  # Concatenated embedding vector (64-dimensional)
```

**Hidden Layers:**
```
h_1 = ReLU(W_1 * h + b_1)  # First hidden layer (64 units)
h_2 = ReLU(W_2 * h_1 + b_2)  # Second hidden layer (32 units)
h_3 = ReLU(W_3 * h_2 + b_3)  # Third hidden layer (16 units)
```

**Output Layer:**
```
ŷ_ui = W_out * h_3 + b_out  # Predicted rating
```

**Loss Function (MSE):**
```
L = (1/N) * Σ (y_ui - ŷ_ui)²
```

### Evaluation Metrics

#### Ranking Metrics

**Precision@K:**
```
Precision@K = |{relevant} ∩ {top-K}| / K
```

**Recall@K:**
```
Recall@K = |{relevant} ∩ {top-K}| / |{relevant}|
```

**NDCG@K:**
```
DCG@K = Σ (2^rel_i - 1) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K
```

**Hit Ratio@K:**
```
Hit@K = 1 if |{relevant} ∩ {top-K}| > 0 else 0
```

#### Rating Metrics

**RMSE:**
```
RMSE = √(Σ (pred_i - actual_i)² / n)
```

**MAE:**
```
MAE = Σ |pred_i - actual_i| / n
```

### Advantages Over Rule-Based Systems

1. **Personalization**: Learns individual user preferences from interaction data
2. **Scalability**: Handles large user and item spaces efficiently
3. **Adaptability**: Continuously improves with more data
4. **Non-linear Patterns**: Captures complex user-item interactions
5. **Cold Start Mitigation**: Uses embeddings to handle new users/items

### Research Contribution

1. **Novel Hybrid Architecture**: Combines disease prediction, nutrition filtering, and NCF
2. **Health-Aware Recommendations**: Ensures food safety for chronic conditions
3. **Personalized Nutrition**: Tailors recommendations to individual preferences
4. **Real-time Inference**: Fast prediction suitable for web applications
5. **Comprehensive Evaluation**: Multiple metrics for thorough assessment

### Novelty Statement

This work presents a novel hybrid recommendation system that integrates:
- **Clinical Decision Support**: Random Forest-based disease prediction
- **Nutritional Safety**: Disease-specific food filtering
- **Deep Learning**: Neural Collaborative Filtering for personalization
- **Holistic Approach**: Considers both health constraints and user preferences

The system is specifically designed for Indian dietary patterns and addresses the unique challenges of nutrition recommendation in healthcare settings.

### Dependencies

- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- Streamlit

### Installation

```bash
pip install tensorflow keras numpy pandas scikit-learn streamlit
```

### Usage Example

```python
from ncf_integration.utils.hybrid_recommender import HybridRecommender

# Initialize recommender
recommender = HybridRecommender(ncf_model_path='ncf_integration/models/ncf_model.keras')

# Get recommendations
result = recommender.recommend(
    user_id=0,
    age=45,
    gender='Male',
    bmi=28.5,
    hba1c=6.8,
    glucose=140,
    sodium=145,
    potassium=4.2,
    bp=130,
    creatinine=1.1,
    top_n=10
)

# Display recommendations
print(f"Detected Diseases: {result['detected_diseases']}")
print(f"Recommendation Method: {result['recommendation_method']}")
print(f"Number of Recommendations: {len(result['recommendations'])}")
```

### Future Enhancements

1. **Side Information Integration**: Incorporate food features into NCF
2. **Multi-Task Learning**: Jointly optimize for rating prediction and ranking
3. **Attention Mechanism**: Focus on important features
4. **Online Learning**: Update model in real-time
5. **Explainability**: Add recommendation explanations
