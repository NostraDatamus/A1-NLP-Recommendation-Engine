"""
===============================================================
Script Name : Main.py
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
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 1.2 Configure Logging
# Purpose: Set up logging to track preprocessing, evaluation, and runtime issues.
# Reasoning/Justification: Enables debugging and ensures transparency in data handling.
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 1.3 Define Global Variables and Styles
# Purpose: Centralise commonly used configurations such as state colours and plotting styles.
# Reasoning/Justification: Ensures consistency across visualisations and standardises outputs.
# ------------------------------------------------------------
sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 10
})

state_colours = {
    'NSW': '#1f77b4',
    'VIC': '#ff7f0e',
    'QLD': '#2ca02c',
    'WA': '#d62728',
    'SA': '#9467bd',
    'TAS': '#8c564b',
    'ACT': '#e377c2',
    'NT': '#7f7f7f'
}

# ------------------------------------------------------------
# 1.4 Load Data
# Purpose: Load the enriched dataset for further processing.
# Reasoning/Justification: Ensures the dataset is available for subsequent steps.
# ------------------------------------------------------------

def load_data(enriched_dataset_path, original_dataset_path):
    try:
        enriched_dataset = pd.read_excel(enriched_dataset_path)
        original_dataset = pd.read_excel(original_dataset_path)
        logger.info("Datasets loaded successfully.")
        return enriched_dataset, original_dataset
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

# print a summary of the dataset
def print_dataset_summary(dataset):
    """ Prints a summary of the dataset. """
    print(f"Dataset Shape: {dataset.shape}")
    print(f"Columns: {dataset.columns}")
    print(f"Missing Values: {dataset.isnull().sum()}")

enriched_dataset_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\enriched_dataset.xlsx'
original_dataset_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\original_dataset.xlsx'
enriched_dataset, original_dataset = load_data(enriched_dataset_path, original_dataset_path)
print_dataset_summary(enriched_dataset)


# ============================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 2.1 Feature Consolidation
# Purpose: Combine API fields into consolidated columns (e.g., Consolidated_Title, Consolidated_Description, Consolidated_Subject_Matter).
# Reasoning/Justification: Simplifies downstream processing by creating consistent, complete features.
# Notes:
# - API fields are consolidated with the following priorities:
#   - Unified_Title: G_Title > O_Title > T_Title
#   - Unified_Description: G_Description > O_Description > T_Description
#   - Unified_Subjects: G_Category > O_Subjects
# - Titles are validated to exclude invalid entries (e.g., 'N/A').
# - Log any changes or dropped entries for traceability.
# ------------------------------------------------------------

# Concatenate the columns 'G_Subtitle', 'O_Subtitle', 'G_Description', 'G_TextSnippet', 'T_Description', 'O_Description', 'T_Summary', 'T_Snippet', 'T_Tags', 'T_Comments', 'O_Excerpts' ,'T_Part Of' into a new column 'Consolidated_Description'
def consolidate_description(row):
    """ Consolidates multiple description fields into a single column. """
    return ' '.join([str(row['G_Subtitle']), str(row['O_Subtitle']), str(row['G_Description']), str(row['G_TextSnippet']), str(row['T_Description']), str(row['O_Description']), str(row['T_Summary']), str(row['T_Snippet']), str(row['T_Tags']), str(row['T_Comments']), str(row['O_Excerpts']), str(row['T_Part Of'])])

enriched_dataset['Consolidated_Description'] = enriched_dataset.apply(consolidate_description, axis=1)

# create new feature 'Consolidated_Subject_Matter' Concatenate the Subject Matter columns 'G_Category' and 'O_Subjects' into a new Feature 'Consolidated_Subject_Matter'
def consolidate_subjects(row):
    """ Consolidates multiple subject fields into a single column. """
    return ' '.join([str(row['G_Category']), str(row['O_Subjects'])])

enriched_dataset['Consolidated_Subject_Matter'] = enriched_dataset.apply(consolidate_subjects, axis=1)

# create new column 'Consolidated_Publisher' by using logic if 'O_Publisher' is not 'N/A' or null then use 'O_Publisher' else if 'G_Publisher' is not 'N/A' or null then use 'G_Publisher' else if 'T_Publisher' is not 'N/A' or null then use 'T_Publisher' else use 'No Text'.
enriched_dataset['Consolidated_Publisher'] = np.where(enriched_dataset['O_Publisher'] != 'N/A', enriched_dataset['O_Publisher'], np.where(enriched_dataset['G_Publisher'] != 'N/A', enriched_dataset['G_Publisher'], np.where(enriched_dataset['T_Publisher'] != 'N/A', enriched_dataset['T_Publisher'], 'No Text')))

# create new column 'Consolidated_Author' by using logic if 'G_Author' is not 'N/A' or null then use 'G_Author' else if 'T_Author' is not 'N/A' or null then use 'T_Author' else if 'O_Author' is not 'N/A' or null then use 'O_Author' else use 'No Text'.
enriched_dataset['Consolidated_Author'] = np.where(enriched_dataset['G_Author'] != 'N/A', enriched_dataset['G_Author'], np.where(enriched_dataset['T_Author'] != 'N/A', enriched_dataset['T_Author'], np.where(enriched_dataset['O_Author'] != 'N/A', enriched_dataset['O_Author'], 'No Text')))

# create new column 'Consolidated_Title' by using logic if 'G_Title' is not 'N/A' or null then use 'G_Title' else if 'O_Title' is not 'N/A' or null then use 'O_Title' else if 'T_Title' is not 'N/A' or null then use 'T_Title' else use 'No Text'.
enriched_dataset['Consolidated_Title'] = np.where(enriched_dataset['G_Title'] != 'N/A', enriched_dataset['G_Title'], np.where(enriched_dataset['O_Title'] != 'N/A', enriched_dataset['O_Title'], np.where(enriched_dataset['T_Title'] != 'N/A', enriched_dataset['T_Title'], 'No Text')))

# ------------------------------------------------------------
# 2.2 Create Published Decade Feature
# Purpose: Extract the year from multiple date columns and assign a refined decade label.
# Reasoning/Justification: Provides a standardised decade-based representation of publishing dates.
# Notes:
# - Extract the latest valid year from 'G_Date', 'T_Date', and 'O_Date' columns. 
# - Assign a refined decade label based on the extracted year.
# - Replace missing years with 'Unknown Publishing Date'.
# ------------------------------------------------------------

# Function to extract the most recent valid year
def extract_year(value):
    """Extracts the most recent valid year from a date or range."""
    if pd.isna(value):
        return None
    try:
        if "-" in value:  # Handle year ranges like "1986-2022"
            return max([int(y) for y in value.split("-") if y.isdigit() and 1500 <= int(y) <= 2024], default=None)
        year = pd.to_datetime(value, errors='coerce').year  # Extract year from full date
        return year if 1500 <= year <= 2024 else None
    except:
        return None

# Consolidate years from G_Date, T_Date, and O_Date
def consolidate_latest_year(row):
    """Consolidates the latest valid year from specific columns."""
    years = [extract_year(row[col]) for col in ['G_Date', 'T_Date', 'O_Date'] if pd.notna(row[col])]
    return max([year for year in years if year is not None], default=None)

# Function to assign published decades
def assign_published_decade(year):
    """Assigns the appropriate decade or category based on year."""
    if pd.isna(year):
        return "Unknown Publishing Date"
    elif year < 1900:
        return "Historical Text"
    else:
        return f"{(year // 10) * 10}s"

# Apply decade assignment
enriched_dataset['Consolidated_Published_Year'] = enriched_dataset.apply(consolidate_latest_year, axis=1)
enriched_dataset['Published_Decade'] = enriched_dataset['Consolidated_Published_Year'].apply(assign_published_decade)

# ------------------------------------------------------------
# 2.3 Create Book Length Feature
# Purpose: Consolidate page count information and assign length categories.
# Reasoning/Justification: Provides a standardised representation of book lengths for comparison.
# Notes:
# - Consolidate page count information from 'G_Page_Count' and 'O_Page_Count' columns. 
# - Assign length categories based on page count ranges.
# - Replace missing page counts with 'Unknown Number of Pages'.
# ------------------------------------------------------------

# Consolidate page counts
enriched_dataset['Consolidated_Page_Count'] = enriched_dataset.apply(
    lambda row: max([val for val in [row['G_Page_Count'], row['O_Page_Count']] if pd.notna(val)], default=None)
    if max([row['G_Page_Count'] or 0, row['O_Page_Count'] or 0]) > 5 else None, axis=1
)

# Define bins and labels
bins = list(range(0, 2200, 100)) + [2143]
labels = [f"{bins[i]} - {bins[i+1]-1} Pages" for i in range(len(bins)-2)] + [f"{bins[-2]}+ Pages"]

# Assign bins to page counts
enriched_dataset['Length'] = pd.cut(
    enriched_dataset['Consolidated_Page_Count'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
).cat.add_categories("Unknown Number of Pages").fillna("Unknown Number of Pages")

# ------------------------------------------------------------
# 2.2 Handle Null Values
# Purpose: Replace missing values in text fields with 'No_Text' and categorical fields with 'Unknown'.
# Reasoning/Justification: Prevents processing errors and standardises missing data handling.
# Notes:
# - 'No_Text' is used as a placeholder for missing text fields to ensure compatibility with text processing steps.
# - 'Unknown' is applied to categorical fields (e.g., Length, Published_Decade).
# ------------------------------------------------------------

print_dataset_summary(enriched_dataset)
enriched_dataset.to_excel('c:\\university\\A1-NLP-Recommendation-Engine\\Outputs\\enriched_dataset_inspection.xlsx', index=False)


# ------------------------------------------------------------
# 2.3 Row Dropping
# Purpose: Remove rows with 5+ columns containing 'No_Text' or 'Unknown'.
# Reasoning/Justification: Improves dataset quality by eliminating records with excessive missing data.
# Notes:
# - Log the number of rows before and after dropping.
# - Calculate and log the percentage of rows dropped.
# ------------------------------------------------------------
# REMOVE ISBNs WITH NO MATCHES WITH ANY API
def drop_invalid_isbns(dataframe, invalid_isbns):
    """ Drops rows with ISBNs that didn't return a match from any of the APIs. """
    return dataframe[~dataframe['ISBN'].astype(str).isin(invalid_isbns)]

# List of invalid ISBNs
invalid_isbns = [
    "9781862512634", "9780730362616", "9798708474995", "9780980309126", "9783600131398",
    "9781488695971", "9780076716623", "9780076855049", "9780078971747", "9781741307641",
    "2019052211324", "9788877610867", "9780076757480", "9780134806631", "9781284142945",
    "9780134637044", "9780136592723", "9780079031808", "9780076775682", "9780730378365",
    "9780547901893", "9780987634559", "9783223221971", "9780190307035", "9780076644667",
    "9780133343168", "9781876703530", "9780987104540", "9780980831573", "9781108720335",
    "9781925505382", "9780655796473", "9781108652339", "9781108603270", "9781108771023",
    "9781108766333", "9780170450508", "9780190306984", "9780076716753", "9781264335794",
    "9780076923441", "9780077023171", "9780190313586", "9780190313654", "9780076819690",
    "9780655780731", "9780077005603", "9780648589624"]

# Drop rows with these ISBNs
original_dataset = drop_invalid_isbns(original_dataset, invalid_isbns)
enriched_dataset = drop_invalid_isbns(enriched_dataset, invalid_isbns)

# Convert ISBNs to strings and strip whitespace
enriched_dataset['ISBN'] = enriched_dataset['ISBN'].astype(str)
enriched_dataset['ISBN'] = enriched_dataset['ISBN'].str.strip()

# Display the updated DataFrames
print(original_dataset.head())
print(enriched_dataset.head())


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
# 4.6 Part-of-Speech (POS) Tagging
# Purpose: Tag parts of speech to add linguistic context for advanced processing tasks.
# Reasoning/Justification: Helps in filtering irrelevant words and improving vocabulary relevance.
# Notes:
# - Example: Retaining nouns and adjectives for content-heavy text analysis.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 4.7 Custom Vocabulary Building
# Purpose: Curate the vocabulary to include domain-specific terms and exclude unnecessary tokens.
# Reasoning/Justification: Aligns text representation with the educational domain and improves recommendation relevance.
# Notes:
# - Ensure curriculum-related terms are prioritised.
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
# 8. BUILDING THE RECOMMENDATION ENGINE
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




































