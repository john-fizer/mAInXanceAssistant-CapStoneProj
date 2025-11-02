# mAInXance Assistant - Predictive Maintenance System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

A machine learning system that predicts maintenance actions from technician problem descriptions using ensemble methods and generates contextual recommendations through retrieval-augmented generation.

---

## Overview

This project addresses a key challenge in manufacturing maintenance: quickly determining the correct maintenance action when equipment fails. The system processes natural language problem descriptions, predicts likely maintenance actions using trained ML models, and generates structured recommendations by retrieving similar historical cases.

**Key Components:**
- Multi-model classification pipeline (6 algorithms evaluated)
- Dimensionality reduction analysis (PCA, t-SNE, UMAP)
- Retrieval-based similar case matching
- LLM-powered narrative generation for actionable guidance

---

## Model Performance

Six classification algorithms were trained and evaluated on maintenance log data:

### Results Summary

| Model | Test Accuracy | Test F1 | CV F1 | Notes |
|-------|--------------|---------|-------|-------|
| **Random Forest** | **94.9%** | **0.97** | **~0.90** | Best overall performance |
| Decision Tree | 94.3% | 0.89 | 0.12 | Severe overfitting detected |
| XGBoost | Trained | - | - | Strong ensemble alternative |
| Logistic Regression | 34.0% | ~0.25 | ~0.25 | Linear model limitation |
| Naive Bayes | 38.1% | ~0.20 | ~0.20 | Struggles with class overlap |
| MLP (Neural Network) | Trained | ~0.22 | ~0.22 | Underperforms on tabular data |
| Baseline (Most Frequent) | 29.2% | 0.05 | - | Reference point |

### Key Findings

**Random Forest selected as production model** due to:
- Highest test accuracy (94.9%)
- Strong F1 score (0.97)
- Stable cross-validation performance
- Effective handling of class imbalance

**Decision Tree overfitting case study:**
Despite 94.3% test accuracy, cross-validation F1 dropped to 0.12, revealing the model memorized training patterns rather than learning generalizable features. This demonstrates the critical importance of cross-validation in model evaluation.

**Linear models and neural networks underperformed** on this TF-IDF feature space with significant class overlap, reinforcing that ensemble methods excel for structured tabular data with engineered features.

---

## System Architecture

### Data Processing Pipeline

1. **Data Cleaning**
   - Text preprocessing (lowercasing, punctuation removal)
   - Tokenization and stopword removal
   - Label mapping to maintenance action categories

2. **Feature Engineering**
   - TF-IDF vectorization of problem descriptions
   - High-dimensional sparse feature representation

3. **Model Training**
   - Stratified train/test split
   - Cross-validation for robust evaluation
   - Hyperparameter tuning (Grid Search for Decision Tree)

4. **Prediction & Generation**
   - Ensemble model inference
   - Similar case retrieval using cosine similarity
   - LLM-powered narrative generation

---

## Dimensionality Reduction Analysis

Three techniques were used to visualize the high-dimensional TF-IDF feature space:

### UMAP
![UMAP Visualization](images/umap.png)

Reveals significant class overlap with some separability. Dense clustering indicates shared technical vocabulary across maintenance actions, while partial separation of certain categories (Electrical Fix, Hydraulic/Pneumatic Fix) suggests learnable patterns.

### PCA
![PCA Visualization](images/PCA_TF-IDF_text_cleaned.png)

Confirms classes are not linearly separable, explaining why linear models achieve only ~34% accuracy. The overlapping variance structures justify the need for non-linear ensemble approaches.

### t-SNE
![t-SNE Visualization](images/t-SNE_2d.png)

Emphasizes local clustering patterns. The massive central cluster with scattered outliers indicates most maintenance problems share common technical language, supporting the effectiveness of similarity-based retrieval.

**Analysis Impact:** These visualizations directly informed model selection strategy and provided intuition about why certain algorithms succeed or fail on this dataset.

---

## Confusion Matrix Analysis

### Random Forest Performance
![Random Forest Confusion Matrix](images/Random_Forest_Heatmap.png)

Strong diagonal indicates excellent classification across most action categories. The model achieves high precision/recall for common maintenance actions (Clean/Clear, Electrical Fix, Hydraulic/Pneumatic Fix, Replace Part, Tighten/Adjust).

### Decision Tree Overfitting
![Decision Tree Confusion Matrix](images/Decision_Tree.png)

Shows strong test performance (94.3% accuracy) but cross-validation revealed F1 of only 0.12, demonstrating how test accuracy alone can be misleading without proper validation.

### MLP Neural Network
![MLP Confusion Matrix](images/MLP_confusion_Matrix.png)

Achieved ~34% accuracy, barely exceeding baseline (29%). Neural networks underperformed due to insufficient data scale, sparse TF-IDF features, and significant class overlap in feature space.

---

## Narrative Generation System

The system goes beyond classification by generating actionable maintenance recommendations.

![Narrative Function](images/Narrative_Function_For_RAG.png)

### Example Output

**Input:**
```
Conveyor stopped, belt squeal, motor running hot near gearbox; intermittent trip on startup.
```

**Generated Output:**

```
— Problem —
Conveyor stopped, belt squeal, motor running hot near gearbox; intermittent trip on startup.

— Top actions —
  • Belt Adjustment: 75.0%
  • Gearbox Inspection: 65.0%
  • Motor Cooling: 45.0%

— Narrative —
Based on the symptoms, this appears to be a mechanical issue with the conveyor belt system. 
The belt squeal and hot motor near the gearbox suggest belt slippage or misalignment. 
First, check if the belt is properly tensioned and aligned to prevent further damage. 
Inspect the gearbox for proper lubrication and signs of wear. 
Plan to clean and lubricate the system, adjust belt tension, and if problems persist, 
consider replacing worn components in the drive system.

— Similar cases —
[1] (Belt Adjustment, similarity=0.92)
    Problem: Conveyor belt slipping and making noise
    Solution: Adjusted tension and realigned belt
    
[2] (Gearbox Repair, similarity=0.85)
    Problem: Motor running hot with grinding noise
    Solution: Replaced worn gears and added lubricant
    
[3] (Motor Cooling, similarity=0.78)
    Problem: Overheating motor causing trips
    Solution: Cleaned vents and improved airflow
```

### Technical Implementation

**Retrieval Component:**
- Cosine similarity search over TF-IDF vectors
- k-nearest neighbors for similar historical cases
- Similarity scoring for confidence ranking

**Generation Component:**
- LLM integration via Groq API (llama3-8b)
- Context assembly from predictions + similar cases
- Structured output formatting for technician use

This approach combines classical ML classification with modern LLM capabilities for interpretable, actionable recommendations.

---

## Installation & Usage

### Prerequisites
- Python 3.9+
- Jupyter Notebook
- Required packages (see requirements.txt)

### Setup

```bash
git clone https://github.com/john-fizer/mAInXanceAssistant-CapStoneProj.git
cd mAInXanceAssistant-CapStoneProj
pip install -r requirements.txt
```

### Running the Notebook

The main analysis is contained in `Main_File_Code_CapStoneProjectMainXance2.ipynb`:

1. Data management and cleaning
2. Label mapping and preprocessing
3. Dimensionality reduction (PCA/t-SNE/UMAP)
4. Model training and comparison
5. Evaluation and analysis
6. Narrative generation system

---

## Technical Stack

**Machine Learning:**
- scikit-learn (classification, vectorization, metrics)
- XGBoost (gradient boosting)
- NumPy, Pandas (data manipulation)

**Natural Language Processing:**
- TF-IDF vectorization
- sentence-transformers (SentenceTransformer)
- Text preprocessing utilities

**Visualization:**
- Matplotlib, Seaborn
- UMAP, t-SNE (dimensionality reduction)

**LLM Integration:**
- Groq API (llama3-8b-8192)
- OpenAI compatible interface

---

## Key Learnings

### Model Selection

**Ensemble methods excel for structured tabular data** - Random Forest and XGBoost significantly outperformed linear models and neural networks on TF-IDF features. Tree-based models naturally capture feature interactions without requiring manual specification.

**Deep learning limitations on small datasets** - MLP achieved only 34% accuracy vs 94.9% for Random Forest. Neural networks require larger datasets and continuous features; tabular data with sparse engineered features favors classical ensemble methods.

**Cross-validation is essential** - Exposed Decision Tree overfitting (94.3% test accuracy but 0.12 CV F1). Single test set evaluation can be misleading, particularly with class imbalance.

### Feature Space Analysis

**Dimensionality reduction guided strategy** - PCA revealed non-linear class boundaries, t-SNE showed local clustering patterns, and UMAP balanced both perspectives. These visualizations explained model performance differences and informed algorithm selection.

**Class overlap explains limitations** - Significant overlap in TF-IDF space means perfect classification is impossible. The ~95% accuracy represents the practical ceiling given vocabulary similarity across maintenance actions.

### System Design

**Retrieval augments classification** - Similar case matching provides validation and context for predictions. Technicians benefit from seeing relevant historical examples alongside model predictions.

**LLM integration adds value** - Structured narrative generation transforms raw predictions into actionable maintenance guidance, making the system practical for real-world use.

---

## Future Enhancements

**Model Improvements:**
- Expand training dataset with more balanced class distribution
- Experiment with ensemble stacking (combining RF + XGBoost)
- Implement active learning for continuous improvement

**Feature Engineering:**
- Sentence embeddings via transformers for better semantic representation
- Domain-specific vocabulary extraction
- Temporal features from maintenance history

**System Enhancements:**
- Deploy as REST API endpoint
- Implement FAISS for scalable similarity search
- Add confidence calibration and uncertainty quantification
- Multi-language support for global manufacturing facilities

---

## Project Structure

```
mAInXanceAssistant-CapStoneProj/
├── Main_File_Code_CapStoneProjectMainXance2.ipynb    # Main notebook
├── images/                                            # Visualizations
│   ├── umap.png
│   ├── PCA_TF-IDF_text_cleaned.png
│   ├── t-SNE_2d.png
│   ├── Random_Forest_Heatmap.png
│   ├── Decision_Tree.png
│   ├── MLP_confusion_Matrix.png
│   ├── ROC.png
│   └── Narrative_Function_For_RAG.png
├── requirements.txt                                   # Dependencies
└── README.md                                          # This file
```

---

## Contact

**John Fizer**  
Email: fizerco@gmail.com  
GitHub: [@john-fizer](https://github.com/john-fizer)

---

## Acknowledgments

- UC Berkeley AI/ML Professional Certification Program
- Modine Manufacturing for domain expertise and problem context
- Open-source ML community (scikit-learn, XGBoost, UMAP)

---

## License

This project was developed as a capstone for UC Berkeley's AI/ML Professional Certification program.
