# MedVeritas Project Flow Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Architecture & File Structure](#project-architecture--file-structure)
3. [Complete Workflow](#complete-workflow)
4. [Notebook Step-by-Step Explanation](#notebook-step-by-step-explanation)
5. [Key Terms & Methods Explained](#key-terms--methods-explained)
6. [Data Flow Diagram](#data-flow-diagram)

---

## Project Overview

**MedVeritas** is a machine learning project that predicts drug effectiveness from patient reviews using Natural Language Processing (NLP) and various ML algorithms. The project processes over 366,000 patient reviews to classify drugs as "effective" or "not effective" and predict exact ratings.

### Main Goals

- **Classification**: Predict if a drug is effective (rating ≥ 7) or not (rating < 7)
- **Regression**: Predict the exact rating (1-10) a patient would give
- **Insights**: Understand what makes a drug effective from patient reviews

---

## Project Architecture & File Structure

### Directory Structure

```
MedVeritas/
│
├── data/
│   ├── raw/                          # Original unprocessed datasets
│   │   ├── drug_ratings.csv
│   │   ├── drugLibTrain_raw.csv
│   │   └── drugsComTrain_raw.csv
│   └── processed/                    # Cleaned and processed data
│       ├── medVe_data_final_version.xlsx    # Main processed dataset
│       └── medVe_data_processed.csv        # CSV version
│
├── notebooks/
│   └── main.ipynb                    # Main analysis notebook (12 steps)
│
├── src/                              # Source code modules
│   ├── data/
│   │   └── processing.py             # Data loading, cleaning, splitting
│   ├── features/
│   │   └── engineering.py            # Feature creation (TF-IDF, encodings)
│   ├── models/
│   │   └── train.py                  # Model training and evaluation
│   ├── nlp/
│   │   └── utils.py                  # NLP utilities (VADER, TextBlob, preprocessing)
│   └── visualization/
│       └── plots.py                  # All plotting functions
│
├── app.py                            # Streamlit interactive dashboard
│
├── models/                           # Saved trained models
│   ├── classification/               # Classification models (.pkl files)
│   └── regression/                   # Regression models (.pkl files)
│
├── results/
│   ├── figures/                      # Generated visualizations (.png)
│   ├── metrics/                      # Model performance metrics (.csv)
│   └── reports/                      # Analysis reports (.md)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## What Each File Does

### 1. **`notebooks/main.ipynb`** - Main Analysis Notebook

**Purpose**: The central notebook that orchestrates the entire analysis pipeline.

**What it does**:

- Imports all necessary modules
- Executes the complete workflow from data loading to model saving
- Generates all visualizations
- Trains and evaluates models
- Saves results and models

**Key Sections**:

1. Data Loading & Cleaning
2. Text Preprocessing & Feature Extraction
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Classification Model Training
7. Regression Model Training
8. Topic Modeling (LDA)
9. NLP Visualizations
10. Insight Visualizations
11. Save Results
12. Summary

---

### 2. **`src/data/processing.py`** - Data Processing Module

**Purpose**: Handles all data loading, cleaning, and preparation tasks.

**Key Functions**:

- **`load_data(file_path)`**:

  - Loads Excel/CSV files into pandas DataFrame
  - Can sample data for testing
  - Returns cleaned DataFrame

- **`clean_data(df)`**:

  - Removes rows with missing reviews, ratings, or drug names
  - Cleans text (removes quotes, special characters)
  - Validates ratings (1-10 range)
  - Removes duplicate reviews
  - Filters reviews shorter than 10 characters
  - Returns cleaned DataFrame

- **`create_effectiveness_label(df, threshold=7)`**:

  - Creates binary label: 1 if rating ≥ 7 (effective), 0 if < 7 (not effective)
  - Adds `is_effective` column to DataFrame

- **`extract_temporal_features(df, date_col)`**:

  - Extracts year, month, year_month from date column
  - Useful for time-series analysis

- **`prepare_train_test_split(df, test_size=0.2)`**:
  - Splits data into 80% training and 20% testing sets
  - Uses stratified sampling to maintain class balance

---

### 3. **`src/nlp/utils.py`** - NLP Utilities Module

**Purpose**: Handles all text preprocessing and sentiment analysis.

**Key Classes & Functions**:

- **`TextPreprocessor`**:

  - **`clean_text(text)`**: Removes URLs, emails, extra spaces
  - **`tokenize(text)`**: Splits text into individual words
  - **`preprocess(text)`**: Full pipeline - cleaning, tokenization, stopword removal, lemmatization

- **`SentimentAnalyzer`**:

  - **`get_vader_sentiment(text)`**: Gets VADER sentiment scores
  - **`get_textblob_sentiment(text)`**: Gets TextBlob sentiment scores
  - **`get_all_sentiment(text)`**: Gets both VADER and TextBlob scores

- **`extract_text_features(df, text_col)`**:

  - Extracts text-based features:
    - Review length (characters)
    - Word count
    - Average word length
    - Exclamation/question mark counts
  - Computes sentiment scores (VADER + TextBlob)
  - Processes in batches for large datasets

- **`preprocess_reviews(df, text_col)`**:
  - Applies full text preprocessing pipeline
  - Creates `review_processed` column with cleaned, lemmatized text

---

### 4. **`src/features/engineering.py`** - Feature Engineering Module

**Purpose**: Creates all features needed for machine learning models.

**Key Functions**:

- **`create_tfidf_features(df, text_col, max_features=500)`**:

  - Converts text to TF-IDF vectors
  - Creates 500 most important word features
  - Returns sparse matrix (memory efficient)

- **`encode_categorical_features(df, categorical_cols)`**:

  - Encodes drug names and conditions as numbers
  - Uses LabelEncoder (each unique value gets a number)
  - Creates `drugName_encoded` and `condition_encoded` columns

- **`create_derived_features(df)`**:

  - **drug_popularity**: How many times each drug appears
  - **condition_popularity**: How many times each condition appears
  - **drug_avg_rating**: Average rating for each drug
  - **condition_avg_rating**: Average rating for each condition
  - **review_length_category**: Categorizes reviews as short/medium/long/very_long

- **`prepare_features(df)`**:
  - Master function that combines all features:
    - Text features (sentiment, length, etc.)
    - TF-IDF features (500 word vectors)
    - Categorical encodings
    - Derived features
  - Returns final feature matrix (520 features total)

---

### 5. **`src/models/train.py`** - Model Training Module

**Purpose**: Trains and evaluates machine learning models.

**Key Class**:

- **`ModelTrainer`**:

  - **`train_classification_models(X_train, y_train, X_test, y_test)`**:

    - Trains 3 models: Logistic Regression, Random Forest, XGBoost
    - Evaluates using: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - Returns results dictionary

  - **`train_regression_models(X_train, y_train, X_test, y_test)`**:

    - Trains 3 models: Linear Regression, Random Forest, XGBoost
    - Evaluates using: RMSE, MAE, R² Score
    - Returns results dictionary

  - **`get_feature_importance(model_name, feature_names)`**:

    - Extracts which features are most important for predictions
    - Returns sorted dictionary of feature importances

  - **`save_model(model_name, model_type, output_dir)`**:

    - Saves trained models as .pkl files

  - **`save_results(output_dir)`**:
    - Saves model performance metrics to CSV files

---

### 6. **`src/visualization/plots.py`** - Visualization Module

**Purpose**: Creates all plots and visualizations.

**Key Functions** (organized by category):

**Statistical Visualizations**:

- `plot_rating_distribution_by_category()`: Box plots and histograms of ratings by drug/condition
- `plot_wordclouds()`: Word clouds for positive vs negative reviews
- `plot_correlation_heatmap()`: Correlation between features and ratings
- `plot_review_length_vs_rating()`: Scatter plot showing relationship

**NLP Visualizations**:

- `plot_topic_modeling_results()`: Visualizes LDA topic modeling results
- `plot_sentiment_distribution_by_condition()`: Sentiment scores across conditions

**Model Evaluation Visualizations**:

- `plot_confusion_matrix()`: Shows model prediction accuracy
- `plot_roc_curves()`: Compares model performance
- `plot_feature_importance()`: Top features that drive predictions

**Insight Visualizations**:

- `plot_top_effective_drugs_by_condition()`: Best drugs for each condition
- `plot_side_effects_by_category()`: Most mentioned side effects
- `plot_time_series_ratings()`: Rating trends over time

---

### 7. **`app.py`** - Streamlit Dashboard

**Purpose**: Interactive web application for exploring the project.

**Features**:

- **Home Page**: Overview statistics
- **Drug Comparison**: Side-by-side comparison of two drugs
- **Effectiveness Predictor**: Input review text → get prediction
- **Data Explorer**: Filter and explore the dataset

**How it works**:

- Loads processed data and trained models
- Uses sentiment analysis for real-time predictions
- Provides interactive visualizations

---

## Complete Workflow

### High-Level Flow

```
1. Data Loading
   ↓
2. Data Cleaning
   ↓
3. Text Preprocessing
   ↓
4. Feature Extraction (Sentiment, TF-IDF, etc.)
   ↓
5. Feature Engineering (Encodings, Derived Features)
   ↓
6. Train-Test Split
   ↓
7. Model Training (Classification & Regression)
   ↓
8. Model Evaluation
   ↓
9. Visualization Generation
   ↓
10. Results Saving
```

### Detailed Data Flow

```
Raw Data (Excel/CSV)
    ↓
[processing.py] load_data()
    ↓
[processing.py] clean_data()
    ├── Remove missing values
    ├── Validate ratings
    ├── Remove duplicates
    └── Clean text
    ↓
[processing.py] create_effectiveness_label()
    └── Add is_effective column (1 if rating ≥ 7)
    ↓
[nlp/utils.py] extract_text_features()
    ├── Calculate review length, word count
    ├── VADER sentiment analysis
    └── TextBlob sentiment analysis
    ↓
[nlp/utils.py] preprocess_reviews()
    ├── Tokenization
    ├── Stopword removal
    └── Lemmatization
    ↓
[features/engineering.py] prepare_features()
    ├── Create TF-IDF features (500 features)
    ├── Encode categorical features
    ├── Create derived features
    └── Combine all features (520 total)
    ↓
[processing.py] prepare_train_test_split()
    ├── 80% Training set
    └── 20% Test set
    ↓
[models/train.py] train_classification_models()
    ├── Logistic Regression
    ├── Random Forest
    └── XGBoost
    ↓
[models/train.py] train_regression_models()
    ├── Linear Regression
    ├── Random Forest
    └── XGBoost
    ↓
[visualization/plots.py] Generate all plots
    ↓
[models/train.py] save_results()
    └── Save models and metrics
```

---

## Notebook Step-by-Step Explanation

### Step 1: Data Loading and Cleaning

```python
# Load data from Excel file
df = load_data('data/processed/medVe_data_final_version.xlsx')

# Clean the data
df = clean_data(df)
# Result: 606,077 → 366,815 rows (removed invalid entries)

# Create effectiveness labels
df = create_effectiveness_label(df, threshold=7)
# Result: 60.2% effective, 39.8% not effective

# Extract temporal features (year, month)
df = extract_temporal_features(df, date_col='date')
```

**What happens**:

- Loads the processed dataset
- Removes invalid entries (missing data, invalid ratings, duplicates)
- Creates binary classification target (effective/not effective)
- Extracts time-based features

---

### Step 2: Text Preprocessing and Feature Extraction

```python
# Extract text features (sentiment, length, etc.)
df = extract_text_features(df, text_col='review')
# Adds: vader_compound, vader_pos, vader_neg, textblob_polarity,
#       review_length, word_count, etc.

# Preprocess reviews (cleaning, tokenization, lemmatization)
df = preprocess_reviews(df, text_col='review', preprocessed_col='review_processed')
# Creates: review_processed column with cleaned text
```

**What happens**:

- Computes sentiment scores for all reviews (VADER + TextBlob)
- Calculates text statistics (length, word count)
- Preprocesses text (removes stopwords, lemmatizes)
- This step is computationally intensive (processes 366K reviews)

---

### Step 3: Exploratory Data Analysis - Statistical Visualizations

```python
# Rating distributions by drug
plot_rating_distribution_by_category(df, category_col='drugName')

# Rating distributions by condition
plot_rating_distribution_by_category(df, category_col='condition')

# Word clouds for positive vs negative reviews
plot_wordclouds(df, text_col='review_processed', rating_col='rating', threshold=7)

# Review length vs rating relationship
plot_review_length_vs_rating(df, length_col='review_length', rating_col='rating')
```

**What happens**:

- Generates visualizations to understand data patterns
- Shows which drugs/conditions have higher ratings
- Identifies common words in positive vs negative reviews
- Analyzes if longer reviews correlate with different ratings

---

### Step 4: Feature Engineering

```python
# Prepare all features for modeling
X, feature_info = prepare_features(
    df,
    text_col='review_processed',
    include_tfidf=True,
    max_tfidf_features=500,
    categorical_cols=['drugName', 'condition']
)
# Result: 520 features total
# - 500 TF-IDF features
# - 20+ other features (sentiment, encodings, derived features)
```

**What happens**:

- Creates TF-IDF vectors from preprocessed text (500 most important words)
- Encodes drug names and conditions as numbers
- Creates derived features (popularity, average ratings)
- Combines all features into single matrix

**Feature Breakdown**:

- **Text Features**: review_length, word_count, vader_compound, textblob_polarity, etc.
- **TF-IDF Features**: 500 word vectors (tfidf_0, tfidf_1, ..., tfidf_499)
- **Categorical Encodings**: drugName_encoded, condition_encoded
- **Derived Features**: drug_popularity, condition_popularity, drug_avg_rating, etc.

---

### Step 5: Train-Test Split

```python
# Split data into training and testing sets
train_df, test_df = prepare_train_test_split(df, test_size=0.2, random_state=42)

# Split features and targets
X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train_class = y_classification.loc[train_indices]  # Classification target
y_test_class = y_classification.loc[test_indices]
y_train_reg = y_regression.loc[train_indices]        # Regression target
y_test_reg = y_regression.loc[test_indices]
```

**What happens**:

- Splits data: 80% training (293,452 rows), 20% testing (73,363 rows)
- Uses stratified sampling to maintain class balance
- Separates features (X) from targets (y)
- Creates separate targets for classification and regression

---

### Step 6: Model Training - Classification

```python
# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train classification models
classification_results = trainer.train_classification_models(
    X_train, y_train_class,
    X_test, y_test_class
)
# Trains: Logistic Regression, Random Forest, XGBoost
# Evaluates: Accuracy, Precision, Recall, F1-Score, ROC-AUC
```

**What happens**:

- Trains 3 different classification models
- Each model learns to predict: Effective (1) or Not Effective (0)
- Evaluates performance on test set
- **Best Model**: XGBoost (78.58% accuracy, 86.28% ROC-AUC)

**Model Comparison**:

- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method, handles non-linear relationships
- **XGBoost**: Gradient boosting, best performance

---

### Step 7: Model Training - Regression

```python
# Train regression models
regression_results = trainer.train_regression_models(
    X_train, y_train_reg,
    X_test, y_test_reg
)
# Trains: Linear Regression, Random Forest, XGBoost
# Evaluates: RMSE, MAE, R² Score
```

**What happens**:

- Trains 3 different regression models
- Each model learns to predict exact rating (1-10)
- Evaluates performance on test set
- **Best Model**: XGBoost (RMSE: 2.63, R²: 0.43)

---

### Step 8: Topic Modeling (LDA)

```python
# Prepare documents for LDA
documents = df['review_processed'].fillna('').astype(str).tolist()
documents = [doc.split() for doc in documents if len(doc.split()) > 5]

# Create dictionary and corpus
dictionary = corpora.Dictionary(documents)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=10
)
```

**What happens**:

- Identifies 10 main topics in the reviews
- Each topic represents a theme (e.g., "side effects", "effectiveness", "time period")
- Helps understand what patients discuss in reviews
- Topics include: side effects, time periods, infections, pain, weight, etc.

---

### Step 9: NLP Visualizations

```python
# Sentiment distribution by condition
plot_sentiment_distribution_by_condition(
    df,
    sentiment_col='vader_compound',
    condition_col='condition'
)
```

**What happens**:

- Shows how sentiment varies across different medical conditions
- Helps identify which conditions have more positive/negative reviews

---

### Step 10: Insight Visualizations

```python
# Top effective drugs by condition
plot_top_effective_drugs_by_condition(df, top_n=10)

# Side effects analysis
plot_side_effects_by_category(df, text_col='review', category_col='drugName')

# Time series ratings
plot_time_series_ratings(df, date_col='date', drug_col='drugName', rating_col='rating')
```

**What happens**:

- Identifies best drugs for each condition
- Analyzes most mentioned side effects
- Shows rating trends over time

---

### Step 11: Save Results

```python
# Save model metrics
trainer.save_results(output_dir='results/metrics')

# Save trained models
for model_name in classification_results.keys():
    trainer.save_model(model_name, 'classification', output_dir='models')

for model_name in regression_results.keys():
    trainer.save_model(model_name, 'regression', output_dir='models')
```

**What happens**:

- Saves model performance metrics to CSV files
- Saves trained models as .pkl files for later use
- Models can be loaded and used for predictions

---

### Step 12: Summary

```python
# Print final summary
print(f"Total reviews: {len(df):,}")
print(f"Unique drugs: {df['drugName'].nunique():,}")
print(f"Classification Models Performance:")
# ... prints all metrics
```

**What happens**:

- Displays final statistics
- Shows model performance summary
- Confirms all outputs are saved

---

## Key Terms & Methods Explained

### NLP (Natural Language Processing) Terms

#### 1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**

**What it is**: A sentiment analysis tool specifically designed for social media text.

**How it works**:

- Uses a dictionary of words with sentiment scores
- Considers word order and punctuation
- Handles negations ("not good" = negative)
- Understands capitalization ("GREAT" = more positive than "great")
- Handles emojis and slang

**Output Scores**:

- **compound**: Overall sentiment score (-1 to +1)
  - Positive (> 0.05): Positive sentiment
  - Neutral (-0.05 to 0.05): Neutral sentiment
  - Negative (< -0.05): Negative sentiment
- **pos**: Positive sentiment score (0 to 1)
- **neu**: Neutral sentiment score (0 to 1)
- **neg**: Negative sentiment score (0 to 1)

**Example**:

```python
text = "This medication is amazing! It really helped me."
vader_scores = {
    'compound': 0.8316,  # Very positive
    'pos': 0.684,
    'neu': 0.316,
    'neg': 0.0
}
```

**Why we use it**: VADER is excellent for short, informal text like patient reviews. It's fast and doesn't require training.

---

#### 2. **TextBlob**

**What it is**: A Python library for processing textual data, including sentiment analysis.

**How it works**:

- Uses pattern-based sentiment analysis
- Provides polarity and subjectivity scores

**Output Scores**:

- **polarity**: Sentiment score (-1 to +1)
  - -1: Very negative
  - 0: Neutral
  - +1: Very positive
- **subjectivity**: How subjective the text is (0 to 1)
  - 0: Objective (factual)
  - 1: Subjective (opinion-based)

**Example**:

```python
text = "This drug worked well for me."
textblob_scores = {
    'polarity': 0.5,      # Moderately positive
    'subjectivity': 0.6    # Somewhat subjective
}
```

**Why we use it**: Provides an alternative sentiment measure to compare with VADER. More subjective/objective information.

---

#### 3. **Tokenization**

**What it is**: Splitting text into individual words (tokens).

**Example**:

```
Input:  "I love this medication!"
Output: ["I", "love", "this", "medication", "!"]
```

**Why we use it**: First step in text processing. Needed for further analysis.

---

#### 4. **Stopwords**

**What it is**: Common words that don't carry much meaning (e.g., "the", "a", "is", "and").

**Why we remove them**:

- Reduces noise in analysis
- Focuses on meaningful words
- Reduces feature space

**Example**:

```
Before: ["I", "love", "this", "medication", "it", "is", "great"]
After:  ["love", "medication", "great"]
```

---

#### 5. **Lemmatization**

**What it is**: Converting words to their base/root form.

**Example**:

```
"running" → "run"
"better"  → "good"
"medications" → "medication"
```

**Why we use it**: Groups related words together. "medication" and "medications" become the same.

---

#### 6. **TF-IDF (Term Frequency-Inverse Document Frequency)**

**What it is**: A numerical statistic that reflects how important a word is to a document in a collection of documents.

**How it works**:

- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare a word is across all documents
- **TF-IDF = TF × IDF**: High score = word is important and unique

**Example**:

```
Word "amazing" appears:
- 10 times in Review A (high TF)
- Rarely in other reviews (high IDF)
→ High TF-IDF score = important feature for Review A

Word "the" appears:
- 50 times in Review A (high TF)
- In almost all reviews (low IDF)
→ Low TF-IDF score = not a useful feature
```

**Why we use it**: Converts text into numerical features that ML models can use. Identifies important words.

**In our project**: We create 500 TF-IDF features (500 most important words across all reviews).

---

#### 7. **LDA (Latent Dirichlet Allocation) - Topic Modeling**

**What it is**: An unsupervised learning method that identifies topics in a collection of documents.

**How it works**:

- Assumes each document is a mixture of topics
- Each topic is a distribution of words
- Discovers hidden topics automatically

**Example**:

```
Topic 1: "Side Effects"
Words: effect (6.6%), side (6.4%), sleep (2.4%), anxiety (1.8%)...

Topic 2: "Time Period"
Words: period (5.9%), month (4.1%), control (3.9%), cramp (2.8%)...

Topic 3: "Pain Relief"
Words: pain (15.2%), relief (1.3%), severe (1.9%)...
```

**Why we use it**: Understands what patients discuss in reviews without reading every review. Identifies themes.

**In our project**: We identify 10 topics from all reviews.

---

### Machine Learning Terms

#### 8. **Classification vs Regression**

**Classification**:

- **Goal**: Predict a category/class
- **Our task**: Predict "Effective" (1) or "Not Effective" (0)
- **Output**: Discrete labels
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Regression**:

- **Goal**: Predict a continuous value
- **Our task**: Predict exact rating (1-10)
- **Output**: Continuous numbers
- **Metrics**: RMSE, MAE, R² Score

---

#### 9. **Train-Test Split**

**What it is**: Dividing data into training and testing sets.

**Why we do it**:

- **Training set**: Used to teach the model
- **Test set**: Used to evaluate how well the model learned
- Prevents overfitting (model memorizing training data)

**Our split**: 80% training, 20% testing

---

#### 10. **Model Types Used**

**Logistic Regression**:

- Linear model for classification
- Fast and interpretable
- Good baseline

**Random Forest**:

- Ensemble of decision trees
- Handles non-linear relationships
- Reduces overfitting

**XGBoost (Extreme Gradient Boosting)**:

- Advanced gradient boosting
- Best performance in our project
- Handles complex patterns

---

#### 11. **Evaluation Metrics**

**Classification Metrics**:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted "effective", how many were actually effective?
- **Recall**: Of actual "effective", how many did we predict correctly?
- **F1-Score**: Balance of precision and recall
- **ROC-AUC**: Area under ROC curve (higher = better, max = 1.0)

**Regression Metrics**:

- **RMSE (Root Mean Squared Error)**: Average prediction error (lower = better)
- **MAE (Mean Absolute Error)**: Average absolute error (lower = better)
- **R² Score**: How well model explains variance (higher = better, max = 1.0)

---

#### 12. **Feature Importance**

**What it is**: Which features (words, sentiment scores, etc.) are most important for predictions.

**Why it matters**: Understands what drives model decisions.

**Example**:

```
Top Features:
1. vader_compound: 0.15    (sentiment is very important)
2. tfidf_42: 0.12           (word "effective" is important)
3. drug_avg_rating: 0.10     (drug's historical rating matters)
```

---

### Data Processing Terms

#### 13. **Label Encoding**

**What it is**: Converting categorical text to numbers.

**Example**:

```
Before:
drugName: ["Aspirin", "Ibuprofen", "Aspirin", "Tylenol"]

After:
drugName_encoded: [0, 1, 0, 2]
```

**Why we use it**: ML models need numbers, not text.

---

#### 14. **Stratified Sampling**

**What it is**: Splitting data while maintaining class proportions.

**Why we use it**: Ensures training and test sets have similar distributions of effective/not effective reviews.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (Excel/CSV)                      │
│             606,077 reviews, 5 columns                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   processing.py: clean_data()  │
        │   - Remove invalid entries    │
        │   - Validate ratings           │
        │   - Remove duplicates          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   CLEANED DATA                 │
        │   366,815 reviews              │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   nlp/utils.py:                │
        │   extract_text_features()      │
        │   - VADER sentiment            │
        │   - TextBlob sentiment         │
        │   - Review length, word count  │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   nlp/utils.py:                │
        │   preprocess_reviews()         │
        │   - Tokenization               │
        │   - Stopword removal           │
        │   - Lemmatization              │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   features/engineering.py:     │
        │   prepare_features()           │
        │   - TF-IDF (500 features)      │
        │   - Categorical encodings       │
        │   - Derived features           │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   FEATURE MATRIX              │
        │   366,815 rows × 520 columns  │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Train-Test Split             │
        │   Train: 293,452 (80%)         │
        │   Test:  73,363 (20%)          │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  CLASSIFICATION  │          │   REGRESSION     │
│  Models:         │          │   Models:        │
│  - Logistic Reg  │          │   - Linear Reg   │
│  - Random Forest │          │   - Random Forest│
│  - XGBoost       │          │   - XGBoost      │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └─────────────┬───────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │   MODEL EVALUATION             │
        │   - Metrics calculated         │
        │   - Visualizations created     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   SAVE RESULTS                │
        │   - Models (.pkl)              │
        │   - Metrics (.csv)             │
        │   - Visualizations (.png)      │
        └───────────────────────────────┘
```

---

## Summary

This project demonstrates a complete machine learning pipeline for analyzing patient reviews:

1. **Data Processing**: Cleaning and preparing 366K+ reviews
2. **NLP**: Extracting sentiment and text features using VADER, TextBlob, TF-IDF
3. **Feature Engineering**: Creating 520 features from text and metadata
4. **Model Training**: Training 6 models (3 classification, 3 regression)
5. **Evaluation**: Achieving 78.58% accuracy in classification
6. **Visualization**: Generating 13+ plots for insights
7. **Deployment**: Interactive dashboard for predictions

The project successfully predicts drug effectiveness from patient reviews and provides actionable insights for healthcare decision-making.

---

## Additional Resources

- **VADER Documentation**: https://github.com/cjhutto/vaderSentiment
- **TextBlob Documentation**: https://textblob.readthedocs.io/
- **TF-IDF Explanation**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- **LDA Topic Modeling**: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

---

_Last Updated: November 2025_
