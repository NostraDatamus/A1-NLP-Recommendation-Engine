"""
===============================================================
Project Name: Assessment 1: NLP Recommendation Engine
Unit Name   : MA5851
Description : This script builds a content-based recommendation engine 
              to suggest textbooks for Australian schools based on year 
              levels, subjects, and other metadata. It involves data 
              collection via APIs, text data wrangling, and machine 
              learning to generate personalized recommendations.
Author      : Alexander Floyd
Student ID  : JC502993
Email       : alex.floyd@my.jcu.edu.au
Date        : December 05, 2024
Version     : 1.0
===============================================================
"""

# ============================================================
# 1. SETUP AND INITIALISATION
# ============================================================

# ------------------------------------------------------------
# 1.1 Import Required Libraries
# Purpose: Load all necessary libraries for data handling, text processing, visualisation, and modelling.
# Reasoning/Justification: Ensures availability of essential tools and modules for all tasks.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1.2 Configure Logging
# Purpose: Set up logging to track preprocessing, evaluation, and runtime issues.
# Reasoning/Justification: Enables debugging and ensures transparency in data handling.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1.3 Define Global Variables and Styles
# Purpose: Centralise commonly used configurations such as state colours and plotting styles.
# Reasoning/Justification: Ensures consistency across visualisations and standardises outputs.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1.4 Load Data
# Purpose: Load the enriched_dataset and original_dataset for further processing.
# Reasoning/Justification: Ensures the datasets are available for subsequent steps.
# ------------------------------------------------------------


# ============================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 2.1 Duplicate Removal
# Purpose: Identify and remove duplicate records (e.g., identical ISBNs).
# Reasoning/Justification: Ensures dataset integrity by removing redundancy.
# Notes:
# - Use ISBN as the primary key for duplicates.
# - In case of ties, prioritise based on additional attributes like Title and Subject.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2.2 Handle Null Values
# Purpose: Replace missing values in text fields with 'No_Text' and categorical fields with 'Unknown'.
# Reasoning/Justification: Prevents processing errors and standardises missing data handling.
# Notes:
# - 'No_Text' is used as a placeholder for missing text fields to ensure compatibility with text processing steps.
# - 'Unknown' is applied to categorical fields (e.g., Length, Published_Decade).
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2.3 Feature Consolidation
# Purpose: Combine API fields into unified columns (e.g., Unified_Title, Unified_Description, Unified_Subjects).
# Reasoning/Justification: Simplifies downstream processing by creating consistent, complete features.
# Notes:
# - API fields are consolidated with the following priorities:
#   - Unified_Title: G_Title > O_Title > T_Title
#   - Unified_Description: G_Description > O_Description > T_Description
#   - Unified_Subjects: G_Category > O_Subjects
# - Titles are validated to exclude invalid entries (e.g., 'N/A').
# - Log any changes or dropped entries for traceability.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2.4 Row Dropping
# Purpose: Remove rows with 5+ columns containing 'No_Text' or 'Unknown'.
# Reasoning/Justification: Improves dataset quality by eliminating records with excessive missing data.
# Notes:
# - Log the number of rows before and after dropping.
# - Calculate and log the percentage of rows dropped.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2.5 Outlier Detection
# Purpose: Detect and address outliers in numerical fields (e.g., Page Count).
# Reasoning/Justification: Improves data quality by mitigating the impact of extreme values.
# Notes:
# - Outliers in Page Count can distort length-based recommendations.
# - Apply thresholds or transformations (e.g., log scaling) as needed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2.6 Standardisation
# Purpose: Ensure numerical fields are standardised (e.g., Length).
# Reasoning/Justification: Normalisation improves comparability and supports similarity calculations.
# Notes:
# - Use min-max scaling or z-score standardisation based on field requirements.
# ------------------------------------------------------------


# ============================================================
# 3. TEXT CLEANING AND NORMALISATION
# ============================================================

# ------------------------------------------------------------
# 3.1 Normalise Text Case
# Purpose: Convert all text to lowercase to standardise text.
# Reasoning/Justification: Removes case-based inconsistencies.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.2 Remove Punctuation and Special Characters
# Purpose: Clean text by removing unnecessary punctuation and symbols.
# Reasoning/Justification: Reduces noise and improves tokenisation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.3 Expand Contractions
# Purpose: Replace contractions with their full forms.
# Reasoning/Justification: Improves tokenisation and text standardisation by avoiding fragmented words.
# Notes:
# - For example, "can't" will be converted to "cannot."
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.4 Fix Encoding Issues
# Purpose: Replace or remove non-standard encoding artifacts (e.g., Foreign character, smart quotes).
# Reasoning/Justification: Prevents processing errors and ensures clean text representation.
# Notes:
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.5 Whitespace Normalisation
# Purpose: Remove excessive whitespace or line breaks.
# Reasoning/Justification: Ensures clean token boundaries and reduces noise in tokenisation.
# Notes:
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.6 Remove Stopwords
# Purpose: Remove frequently occurring but semantically insignificant words (e.g., 'and', 'the').
# Reasoning/Justification: Improves text representation by focusing on meaningful words.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.7 Apply Stemming or Lemmatization
# Purpose: Reduce words to their root form.
# Reasoning/Justification: Normalises text variations for better similarity calculations.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.8 Corpus Validation
# Purpose: Validate the cleaned corpus to ensure no invalid placeholders (e.g., 'No_Text') remain.
# Reasoning/Justification: Ensures the quality of the text data before further processing.
# ------------------------------------------------------------


# ============================================================
# 3. TEXT CLEANING AND NORMALISATION
# ============================================================

# ------------------------------------------------------------
# 3.1 Normalise Text Case
# Purpose: Convert all text to lowercase to standardise text.
# Reasoning/Justification: Removes case-based inconsistencies.
# Notes:
# - Standardising the case ensures consistent comparisons during similarity calculations.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.2 Remove Punctuation and Special Characters
# Purpose: Clean text by removing unnecessary punctuation and symbols.
# Reasoning/Justification: Reduces noise and improves tokenisation.
# Notes:
# - Symbols that do not contribute to semantic meaning, such as commas and periods, are removed.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.3 Expand Contractions
# Purpose: Replace contractions with their full forms.
# Reasoning/Justification: Improves tokenisation and text standardisation by avoiding fragmented words.
# Notes:
# - For example, "can't" will be converted to "cannot."
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.4 Fix Encoding Issues
# Purpose: Replace or remove non-standard encoding artifacts (e.g., smart quotes).
# Reasoning/Justification: Prevents processing errors and ensures clean text representation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.5 Whitespace Normalisation
# Purpose: Remove excessive whitespace or line breaks.
# Reasoning/Justification: Ensures clean token boundaries and reduces noise in tokenisation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.6 Remove Stopwords
# Purpose: Remove frequently occurring but semantically insignificant words (e.g., 'and', 'the').
# Reasoning/Justification: Improves text representation by focusing on meaningful words.
# Notes:
# - Custom stopword lists may be added for project-specific noise words.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.7 Apply Stemming or Lemmatisation
# Purpose: Reduce words to their root or base forms.
# Reasoning/Justification: Normalises text variations for better similarity calculations.
# Notes:
# - Lemmatisation is preferred for retaining semantic meaning.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3.8 Retain Numbers
# Purpose: Retain numeric data in the text corpus.
# Reasoning/Justification: Numbers are often semantically significant in educational texts (e.g., grade levels).
# Notes:
# - Ensure that numbers are not removed during cleaning.
# ------------------------------------------------------------


# ============================================================
# 4. TEXT REPRESENTATION AND TOKENISATION
# ============================================================


# ------------------------------------------------------------
# 4.1 Sentence Tokenisation
# Purpose: Split text into sentences to allow higher-level structural analysis.
# Reasoning/Justification: Enables more granular processing and the identification of contextually relevant units.
# Notes:
# - Sentence tokenisation precedes word tokenisation to ensure proper segmentation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4.2 Word Tokenisation
# Purpose: Split sentences into individual words or tokens for further processing.
# Reasoning/Justification: Provides the foundational structure for text vectorisation and similarity calculations.
# Notes:
# - Retains numerics and tokens relevant to the domain.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4.3 Named Entity Recognition (NER)
# Purpose: Identify named entities (e.g., person, organisation, location) in the text.
# Reasoning/Justification: Provides additional contextual features for downstream tasks.
# Notes:
# - Use pre-trained models like spaCy for NER.
# - Entities can include educational terms, locations, and institutions.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4.4 Bigram/Trigram Extraction
# Purpose: Identify multi-word phrases to enhance contextual representation.
# Reasoning/Justification: Captures semantic meaning by identifying commonly co-occurring terms (e.g., "data science").
# Notes:
# - Use statistical measures like PMI to filter meaningful bigrams/trigrams.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4.5 Domain-Specific Phrase Extraction
# Purpose: Extract domain-specific phrases relevant to the educational context (e.g., "grade 7 curriculum").
# Reasoning/Justification: Enhances the representation of domain-relevant phrases for better similarity calculations.
# Notes:
# - Combine results from bigram/trigram extraction with domain-specific knowledge.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 3.4 Corpus Validation
# Purpose: Validate the cleaned corpus to ensure no invalid placeholders (e.g., 'No_Text') remain.
# Reasoning/Justification: Ensures the quality of the text data before further processing.
# ------------------------------------------------------------



# ============================================================
# 5. BUILDING THE DOCUMENT MATRIX
# ============================================================

# ------------------------------------------------------------
# 5.1 Create Term Frequency-Inverse Document Frequency (TF-IDF) Matrix
# Purpose: Represent text numerically based on term importance.
# Reasoning/Justification: Captures the relevance of terms while reducing the impact of common words.
# Notes:
# - TF-IDF is suitable for content-based recommendation systems due to its focus on term importance.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5.2 Generate Document Term Matrix
# Purpose: Create the TF-IDF matrix using the tokenised text.
# Reasoning/Justification: Provides a numerical representation of text for similarity calculations.
# Notes:
# - Ensure input text uses the cleaned and tokenised corpus.
# - Retain alignment with the custom vocabulary created earlier.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5.3 Inspect Sparsity
# Purpose: Evaluate the sparsity of the TF-IDF matrix.
# Reasoning/Justification: Identifies potential issues with high-dimensional sparse data.
# Notes:
# - If sparsity is too high, consider dimensionality reduction techniques.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5.4 Apply Dimensionality Reduction (if necessary)
# Purpose: Reduce the dimensionality of the document matrix.
# Reasoning/Justification: Improves computational efficiency without sacrificing relevance.
# Notes:
# - Techniques like Singular Value Decomposition (SVD) may be applied.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5.5 Save Document Matrix
# Purpose: Save the TF-IDF matrix for reproducibility and downstream tasks.
# Reasoning/Justification: Ensures the matrix can be reused without reprocessing.
# Notes:
# - Use a suitable file format such as `.npz` for sparse matrices.
# ------------------------------------------------------------


# ============================================================
# 6. SIMILARITY COMPUTATION
# ============================================================

# ------------------------------------------------------------
# 6.1 Compute Cosine Similarity
# Purpose: Calculate text similarity using cosine similarity on the TF-IDF matrix.
# Reasoning/Justification: Captures the semantic similarity between documents in a high-dimensional space.
# Notes:
# - Cosine similarity is computationally efficient and suitable for sparse matrices.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6.2 Apply Adjacent Year Group Weighting
# Purpose: Reduce similarity scores for adjacent year groups by 50%.
# Reasoning/Justification: Maintains relevance while accounting for natural variance in the dataset.
# Notes:
# - Adjacent year groups receive lower weight compared to the actual year group.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6.3 Weight Similarity Features
# Purpose: Combine multiple similarity scores (e.g., text, subject, year group) into a single weighted score.
# Reasoning/Justification: Balances the importance of different features in generating recommendations.
# Notes:
# - Support configurable weights for each feature to customise recommendations.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6.4 Normalise Similarity Scores
# Purpose: Scale all similarity scores to a common range (e.g., 0-1).
# Reasoning/Justification: Ensures interpretability and consistency when combining multiple similarity metrics.
# Notes:
# - Normalisation aids in effectively combining scores from different features.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6.5 Filter by Relevance Threshold
# Purpose: Exclude items below a predefined similarity threshold.
# Reasoning/Justification: Ensures the quality of recommendations by filtering out irrelevant results.
# Notes:
# - The threshold value should be adjustable based on the use case.
# ------------------------------------------------------------


# ============================================================
# 7. CORPUS EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 7.1 Corpus Summary and Statistics
# Purpose: Summarise key statistics of the corpus to understand its overall structure.
# Reasoning/Justification: Provides an overview of corpus characteristics such as size, content, and feature distribution.
# Notes:
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.2 Document Metrics
# Purpose: Calculate metrics for evaluating the corpus structure and its alignment with project objectives.
# Reasoning/Justification: Provides insights into corpus characteristics, ensuring completeness and relevance.
# Notes:
# - Metrics to include:
#   - Total Documents
#   - Total Tokens
#   - Total Phrases
#   - Total Entries
#   - Average Tokens per Document
#   - Average Phrases per Document
#   - Average Entries per Document
#   - Average Phrases per Entry
#   - Average Tokens per Entry
#   - Average Tokens per Phrase
#   - Vocabulary of Target Tokens
#   - Out of Vocabulary Tokens
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.3 Visualise Term Frequencies
# Purpose: Identify the most and least common terms in the corpus.
# Reasoning/Justification: Highlights dominant terms and informs feature selection.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.4 Inspect Class Distributions
# Purpose: Examine distributions across target labels (e.g., Subject).
# Reasoning/Justification: Identifies potential imbalances.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.5 Explore Token Distributions
# Purpose: Analyse token frequency, average token length, and other metrics to understand corpus characteristics.
# Reasoning/Justification: Identifies potential issues like excessive sparsity or irrelevant terms.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.6 Analyse Corpus Sparsity
# Purpose: Evaluate the sparsity of the document matrix to identify gaps in representation.
# Reasoning/Justification: Ensures the matrix is neither overly sparse nor dense.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.7 Validate Vocabulary Coverage
# Purpose: Ensure the vocabulary effectively covers the domain-specific terms and corpus content.
# Reasoning/Justification: Aligns the vocabulary with the recommendation engine's focus areas.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 7.8 Identify Outliers
# Purpose: Detect anomalous entries in the corpus or matrix that may skew analysis.
# Reasoning/Justification: Prevents outliers from disproportionately influencing the results.
# ------------------------------------------------------------


# ============================================================
# 8. BUILDING THE RECOMMENDATION ENGINE & PROFILES
# ============================================================

# ------------------------------------------------------------
# 8.1 Input Validation
# Purpose: Validate user inputs to ensure sufficient data for recommendation generation.
# Reasoning/Justification: Prevents errors by ensuring required inputs are provided.
# Notes:
# - Ensure at least two of the three inputs (ISBN, Subject, Year Group) are given.
# - Prompt users for missing inputs if necessary.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.2 Compute Combined Similarity Scores
# Purpose: Integrate text, subject, and year group similarity scores using predefined or optimised weights.
# Reasoning/Justification: Ensures all relevant features are considered for personalised recommendations.
# Notes:
# - Normalise and combine similarity scores for interpretability.
# - Adjust weights for features based on domain relevance.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.3 Apply Adjacent Year Group Weighting
# Purpose: Adjust similarity scores for adjacent year groups by 50%.
# Reasoning/Justification: Ensures relevance while accounting for natural variance in educational levels.
# Notes:
# - Prioritise actual year groups over adjacent groups.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.4 Filter Recommendations
# Purpose: Exclude items below a predefined relevance threshold.
# Reasoning/Justification: Ensures only high-quality recommendations are presented.
# Notes:
# - Make the threshold configurable.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.5 Rank Recommendations
# Purpose: Rank books by combined similarity scores.
# Reasoning/Justification: Delivers the most relevant recommendations to the user.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.6 Handle Insufficient Recommendations
# Purpose: Return as many recommendations as possible if the top-N threshold is not met.
# Reasoning/Justification: Ensures usability even with limited data.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.7 Format Recommendations
# Purpose: Prepare output to display Book Title, Subject, Length, Published Decade, and similarity score.
# Reasoning/Justification: Ensures outputs are user-friendly and informative.
# Notes:
# - Include a star rating based on normalised similarity scores (0-5).
# ------------------------------------------------------------

# ------------------------------------------------------------
# 8.8 Log Recommendations
# Purpose: Document inputs, outputs, and decision-making details for traceability.
# Reasoning/Justification: Ensures transparency and aids debugging.
# Notes:
# - Log to both console and a dedicated file.
# ------------------------------------------------------------


# ============================================================
# 9. HYPERTUNING, VALIDATION AND OPTIMISATION
# ============================================================

# ------------------------------------------------------------
# 9.1 Define Hyperparameter Search Space
# Purpose: Specify ranges for similarity weights, relevance thresholds, and other tunable parameters.
# Reasoning/Justification: Ensures systematic exploration of parameter combinations.
# Notes:
# - Include predefined ranges based on domain knowledge.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.2 Perform Grid Search
# Purpose: Optimise similarity weights for best performance.
# Reasoning/Justification: Improves recommendation relevance by fine-tuning parameters.
# Notes:
# - Use cross-validation to evaluate parameter effectiveness.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.3 Cross-Validation
# Purpose: Evaluate engine performance across multiple folds of the dataset.
# Reasoning/Justification: Ensures robustness and prevents overfitting.
# Notes:
# - Implement k-fold cross-validation with a suitable value for k.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.4 Evaluate Metrics
# Purpose: Calculate Precision, Recall, F1 Score, ROC AUC, Top-10 Accuracy, and MRR.
# Reasoning/Justification: Measures engine effectiveness quantitatively.
# Notes:
# - Use "Subject" as the primary label for evaluation.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.5 Visualise Hyperparameter Tuning Results
# Purpose: Generate plots to analyse the effects of hyperparameters on evaluation metrics.
# Reasoning/Justification: Highlights optimal parameter ranges and their impact on performance.
# Notes:
# - Include metrics such as Precision, Recall, F1, and MRR.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.6 Normalise Evaluation Metrics
# Purpose: Standardise metrics to a common scale for easier comparison.
# Reasoning/Justification: Improves interpretability and ensures consistent reporting.
# Notes:
# - Scale metrics to a 0-1 range or other relevant normalisation approach.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 9.7 Log Optimisation Process
# Purpose: Document parameters, scores, and results for reproducibility.
# Reasoning/Justification: Ensures transparency and facilitates debugging.
# Notes:
# - Log details to both console and a dedicated file.
# ------------------------------------------------------------


# ============================================================
# 10. DEMONSTRATING THE RECOMMENDATION ENGINE
# ============================================================

# ------------------------------------------------------------
# 10.1 Setup Demonstration Scenario
# Purpose: Define a practical use case for the recommendation engine.
# Reasoning/Justification: Validates engine functionality in context.
# Notes:
# - Example scenario: A new school requires book recommendations for a specific subject and year group.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.2 Validate Demonstration Inputs
# Purpose: Ensure that the required inputs (ISBN, Subject, Year Group) are sufficient.
# Reasoning/Justification: Prevents errors due to missing or invalid input.
# Notes:
# - Prompt the user for additional input if necessary.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.3 Generate and Rank Recommendations
# Purpose: Use the recommendation engine to produce and rank a list of books based on similarity scores.
# Reasoning/Justification: Demonstrates the engine's ability to generate accurate and relevant suggestions.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.4 Handle Insufficient Recommendations
# Purpose: Return as many recommendations as possible if the top-N threshold is not met.
# Reasoning/Justification: Ensures usability even when the dataset is limited.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.5 Display Recommendations
# Purpose: Present the top recommendations in a user-friendly format.
# Reasoning/Justification: Enhances usability and facilitates interpretation.
# Notes:
# - Display Book Title, Subject, Length, Published Decade, similarity score, and star ratings.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.6 Visualise Evaluation Metrics
# Purpose: Summarise engine performance with plots.
# Reasoning/Justification: Provides a clear, visual representation of results.
# Notes:
# - Include metrics such as Precision, Recall, F1 Score, and MRR.
# - Use "state_colours" for state-based visualisations.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 10.7 Save Demonstration Outputs
# Purpose: Log demonstration results for documentation and reporting purposes.
# Reasoning/Justification: Ensures reproducibility and traceability.
# Notes:
# - Save inputs, outputs, and results to a log file.
# ------------------------------------------------------------