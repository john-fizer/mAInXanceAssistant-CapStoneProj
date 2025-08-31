# mAInXanceAssistant-CapStoneProj
Capstone Project Report – mAInXance Assistant

1. Project Overview

This project builds a predictive maintenance assistant that takes technician notes or problem descriptions and suggests the most likely maintenance actions. The workflow covers data management, model training, performance evaluation, and a narrative conversion step that makes predictions interpretable and useful for end users.

The goal is to demonstrate both technical knowledge of machine learning pipelines and practical application for a diagnostic assistant.

2. Data Management

Cleaning: Text descriptions were preprocessed (tokenization, lowercasing, removing stopwords) and mapped to labeled maintenance actions.

Labeling: Actions were categorized into classes such as Tighten/Adjust, Electrical Fix, Replace Part, etc.

3. Dimensionality Reduction and Visualizations

Dimensionality reduction was applied to visualize high-dimensional TF-IDF features. PCA showed overlapping variance structures, confirming global class overlap. t-SNE emphasized local neighborhood clusters, revealing that some categories (e.g., Electrical Fix vs Hydraulic Fix) are separable but others blend heavily. UMAP balanced these perspectives, showing partial grouping with significant overlap. These plots confirm that classification is challenging: the classes are not linearly separable, requiring more advanced models like ensembles or gradient boosting.

PCA: Showed overlapping class clusters, confirming the dataset is not perfectly separable.

t-SNE: Produced tighter local clusters, highlighting similarities within action categories.

UMAP: Preserved global structure, showing stronger grouping than PCA.

Interpretation: These plots confirm that while patterns exist, there is inherent overlap between classes. This explains why accuracy does not reach 100%.

4. Model Training and Testing
Baseline

Accuracy: ~29%

Macro F1: ~0.05

Serves as the benchmark. All models must outperform this to add value.

Results Summary (Hold-Out Test Set)
Model	Accuracy	Macro F1	Notes
Logistic Regression	~0.34	~0.25	Weak on minority classes
Naive Bayes	~0.38	~0.20	Slightly better, but still weak
Decision Tree	~0.94 (test)	~0.88 (test) / 0.12 (CV)	Overfitting
Random Forest	~0.94–0.95	Strong balance	More stable than single tree
MLP (Neural Net)	~0.34	~0.22	Underfit, needs tuning
XGBoost	~0.81	~0.81	Best trade-off, strong across classes
5. Overfitting and Generalization

Most overfitting: Decision Tree (train ≈ 1.0 vs CV F1 ≈ 0.12).

Most underfitting: Naive Bayes (fails to capture complex interactions).

Best balance: Random Forest and XGBoost.

Neural Net (MLP): Did not overfit heavily, but underperformed relative to ensembles.

Cross-validation was critical here: without it, the Decision Tree would have looked “perfect” when it was just memorizing the training set.

6. Narrative Conversion

Raw predictions are not directly useful to technicians. To address this, a narrative generation step was introduced:

Combines predicted action probabilities with retrieval of similar past cases.

Produces short, professional summaries of:

What’s most likely going on.

What to check first.

Recommended next actions.

Alternate backup steps.

Example

Input: “Conveyor stopped, belt squeal, motor hot near gearbox.”

Output: Narrative covering likely cause, first checks, action plan, and backup steps.

This turns raw classification into a practical maintenance guide.

Actual Output Example: 

— Problem —
Conveyor stopped, belt squeal, motor running hot near gearbox; intermittent trip on startup.

— Top actions —
  • Belt Adjustment: 75.0%
  • Gearbox Inspection: 65.0%
  • Motor Cooling: 45.0%

— Narrative —
Based on the symptoms, this appears to be a mechanical issue with the conveyor belt system. The belt squeal and hot motor near the gearbox suggest belt slippage or misalignment. First, check if the belt is properly tensioned and aligned to prevent further damage. Inspect the gearbox for proper lubrication and signs of wear. Plan to clean and lubricate the system, adjust belt tension, and if problems persist, consider replacing worn components in the drive system.

— Similar cases —
[1] (Belt Adjustment, sim=0.92)
    Desc : Conveyor belt slipping and making noise
    Notes: Adjusted tension and realigned belt
[2] (Gearbox Repair, sim=0.85)
    Desc : Motor running hot with grinding noise
    Notes: Replaced worn gears and added lubricant
[3] (Motor Cooling, sim=0.78)
    Desc : Overheating motor causing trips
    Notes: Cleaned vents and improved airflow

7. Key Findings and Lessons Learned

Not all models are equal: ensembles (RF/XGBoost) outperformed deep learning (MLP) on this dataset.

Visualization confirmed that classes overlap, which explains misclassifications.

Cross-validation was essential to expose overfitting.

Narrative conversion makes the system usable, not just technical.

8. Next Steps

Expand dataset with more balanced class distribution.

Explore model tuning for MLP with more compute resources.

Integrate Retrieval-Augmented Generation (RAG) into the narrative function.

Deploy Random Forest or XGBoost as the backbone model, with narrative conversion layered on top.

Test on fresh, unseen maintenance logs to confirm generalization.

9. How to Run
Prerequisites

Python 3.9+

Jupyter Notebook / Google Colab

Install dependencies:

pip install -r requirements.txt

Steps

Clone repository / open notebook in Colab.

Run notebooks in order:

01_data_preprocessing.ipynb → Clean and prep the data.

02_model_training.ipynb → Train and evaluate ML models.

03_model_evaluation.ipynb → Generate metrics, plots, and comparisons.

04_narrative_assistant.ipynb → Turn predictions into technician narratives.

Load pre-trained artifacts (saved TF-IDF vectorizer, label encoder, models).

Test inference with a new description:

Input: “Hydraulic cylinder leaking near fitting”

Output: Predicted label (Hydraulic/Pneumatic Fix) + Narrative recommendation.
