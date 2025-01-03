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
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
from contractions import fix
from nltk.corpus import stopwords
import nltk
import unicodedata
from contractions import fix
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz, process
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import json
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed


# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_trf")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialise the lemmatiser and stop words
lemmatiser = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

# Set the default plotting style
sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 10
})

# Define state colours for visualisations
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

original_dataset_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\original_dataset.xlsx'
enriched_dataset_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\enriched_dataset.xlsx'
cleaned_dataset_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\cleaned_dataset.xlsx'
tokenised_corpus_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\tokenised_corpus.csv'

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

# Consolidate Description information into a single feature 'Consolidated_Description'
def consolidate_description(row):
    """Consolidates multiple description fields into a single column, ignoring NaN values."""
    return ' '.join([str(row[col]) for col in [
        'G_Subtitle', 'O_Subtitle', 'G_Description', 'G_TextSnippet', 
        'T_Description', 'O_Description', 'T_Summary', 'T_Snippet', 
        'T_Tags', 'T_Comments', 'O_Excerpts', 'T_Part Of'
    ] if pd.notna(row[col])]).strip()  # Remove leading/trailing whitespace

# Apply the function
enriched_dataset['Consolidated_Description'] = enriched_dataset.apply(consolidate_description, axis=1)

# create new feature 'Consolidated_Subject_Matter' Concatenate the Subject Matter columns 'G_Category' and 'O_Subjects' into a new Feature 'Consolidated_Subject_Matter'
def consolidate_subjects(row):
    """Consolidates multiple subject fields into a single column, ignoring NaN values."""
    return ' '.join([str(row[col]) for col in ['G_Category', 'O_Subjects'] if pd.notna(row[col])]).strip()

# Apply the function
enriched_dataset['Consolidated_Subject_Matter'] = enriched_dataset.apply(consolidate_subjects, axis=1)

# Consolidate Publisher information into a single column 'Consolidated_Publisher'
def consolidate_publisher(row):
    for col in ['O_Publisher', 'G_Publisher', 'T_Publisher']:
        if pd.notna(row[col]) and row[col] != 'N/A' and row[col] != '0':
            return row[col]
    return 'No_Text'

# Apply the function to create 'Consolidated_Publisher'
enriched_dataset['Consolidated_Publisher'] = enriched_dataset.apply(consolidate_publisher, axis=1)

# create new column 'Consolidated_Author' by using logic if 'G_Author' is not 'N/A' or null then use 'G_Author' else if 'T_Author' is not 'N/A' or null then use 'T_Author' else if 'O_Author' is not 'N/A' or null then use 'O_Author' else use 'No Text'.
enriched_dataset['Consolidated_Author'] = np.where(enriched_dataset['G_Author'] != 'N/A', enriched_dataset['G_Author'], np.where(enriched_dataset['T_Author'] != 'N/A', enriched_dataset['T_Author'], np.where(enriched_dataset['O_Author'] != 'N/A', enriched_dataset['O_Author'], 'No_Text')))

# create new column 'Consolidated_Title' by using logic if 'G_Title' is not 'N/A' or null then use 'G_Title' else if 'O_Title' is not 'N/A' or null then use 'O_Title' else if 'T_Title' is not 'N/A' or null then use 'T_Title' else use 'No Text'.
enriched_dataset['Consolidated_Title'] = np.where(enriched_dataset['G_Title'] != 'N/A', enriched_dataset['G_Title'], np.where(enriched_dataset['O_Title'] != 'N/A', enriched_dataset['O_Title'], np.where(enriched_dataset['T_Title'] != 'N/A', enriched_dataset['T_Title'], 'No_Text')))
# Duplicate the book title for later use in the recommendation engine.
enriched_dataset['Book Title'] = enriched_dataset['Consolidated_Title']
# Replace missing titles with 'Unknown Title'
enriched_dataset['Book Title'] = enriched_dataset['Book Title'].fillna('Unknown Title')

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
        return f"{int((year // 10) * 10)}s"  # Ensure no decimals in output


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
# 2.2 Data Cleaning
# Purpose: Clean the dataset by handling missing values and removing unessessary data.
# Reasoning/Justification: Prevents processing errors and standardises missing data handling.
# Notes:
# - 'No_Text' is used as a placeholder for missing text fields to ensure compatibility with text processing steps.
# - Rows with missing API data are dropped to maintain data integrity.
# - Unused features are removed to streamline the dataset.
# ------------------------------------------------------------

# Replace 'No_Text' with empty strings
enriched_dataset[['Consolidated_Description', 'Consolidated_Subject_Matter', 'Consolidated_Publisher', 'Consolidated_Author', 'Consolidated_Title']] = enriched_dataset[['Consolidated_Description', 'Consolidated_Subject_Matter', 'Consolidated_Publisher', 'Consolidated_Author', 'Consolidated_Title']].replace('No_Text', '')


# Drop rows that have no API data and print number dropped
rows_to_drop = enriched_dataset[['G_Title', 'O_Title', 'T_Title']].replace('', np.nan).isna().all(axis=1)
enriched_dataset = enriched_dataset[~rows_to_drop]

print(f"Number of rows dropped: {rows_to_drop.sum()}")
print(f"Remaining rows: {len(enriched_dataset)}")

# Drop unused features
enriched_dataset = enriched_dataset.drop(columns=['G_Title', 'G_Subtitle', 'G_Author', 'G_Publisher', 'G_Date', 'G_Description', 'G_Category', 'G_Page_Count', 'G_TextSnippet', 'T_Title', 'T_Author', 'T_Publisher', 'T_Date', 'T_Description', 'T_Tags', 'T_Comments', 'T_Part Of', 'T_Summary', 'T_Snippet', 'O_Title', 'O_Subtitle', 'O_Author', 'O_Publisher', 'O_Date', 'O_Description', 'O_Excerpts', 'O_Subjects', 'O_Page_Count', 'Consolidated_Published_Year', 'Consolidated_Page_Count'])

# Save the cleaned dataset
cleaned_dataset = enriched_dataset.copy()
print_dataset_summary(cleaned_dataset)
#cleaned_dataset.to_excel(cleaned_dataset_path, index=False)
cleaned_dataset.to_csv(cleaned_dataset_path, index=False)


# ============================================================
# 3. TEXT CLEANING AND NORMALISATION
# ============================================================

# ------------------------------------------------------------
# 3.1 Custom Vocabulary & Stop Words Building
# Purpose: Load custom vocabulary and stop words list for text processing.
# Reasoning/Justification: Enhances text processing by incorporating domain-specific terms and removing irrelevant words.
# - Ensure curriculum-related terms are prioritised.
# ------------------------------------------------------------

# load Custom_Vocab.py file
# Import custom vocabulary lists
from Custom_Vocab import (
    science_vocab, 
    mathematics_vocab, 
    english_vocab, 
    performance_arts_vocab, 
    practical_studies_vocab, 
    business_and_law_vocab, 
    design_and_technology_vocab, 
    environmental_science_vocab, 
    humanities_vocab, 
    languages_vocab
)

# ------------------------------------------------------------
# 3.2 Text Cleaning
# Purpose: Convert all text to lowercase to standardise text.
# Reasoning/Justification: Removes case-based inconsistencies.
# Notes:
# - Normalise Text Case ensures consistent comparisons during similarity calculations.
# - Removed Punctuation and Special Characters to clean text by removing unnecessary punctuation and symbols.
# - Expand Contractions to replace contractions with their full forms.
# - Fix Encoding Issues to replace or remove non-standard encoding artifacts (e.g., Foreign character, smart quotes).
# - Whitespace Normalisation to remove excessive whitespace or line breaks.
# - Remove Stopwords to eliminate common words that do not contribute to the meaning.
# - Lemmatisation to reduce words to their base or root form.
# ------------------------------------------------------------



# Combined function for text preprocessing
def preprocess_text(text):
    if pd.isna(text):  # Skip NaN values
        return text
    
    # Expand contractions (e.g., "can't" -> "cannot")
    text = fix(text)
    
    # Normalise Unicode and remove non-Latin characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation and symbols except hyphens, apostrophes, and numbers
    text = re.sub(r"[^a-z0-9\s\-']", ' ', text)
    
    # Replace multiple spaces and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenise, remove stopwords, and lemmatise words
    words = text.split()
    cleaned_words = [lemmatiser.lemmatize(word) for word in words if word not in stop_words]
    
    # Rejoin into a cleaned string
    return ' '.join(cleaned_words)

# List of text columns to process
text_columns = [
    'Consolidated_Description', 'Consolidated_Subject_Matter',
    'Consolidated_Publisher', 'Consolidated_Author', 'Consolidated_Title'
]

# Replace 'No Text' with NaN
for col in text_columns:
    cleaned_dataset[col] = cleaned_dataset[col].replace('No_Text', pd.NA)

# Apply the text preprocessing function
for col in text_columns:
    cleaned_dataset[col] = cleaned_dataset[col].apply(preprocess_text)

# Replace NaN back with 'No Text'
for col in text_columns:
    cleaned_dataset[col] = cleaned_dataset[col].fillna('No_Text')

#export the cleaned dataset
cleaned_dataset.to_excel(cleaned_dataset_path, index=False)
print("Cleaned Dataset saved successfully.")

# ------------------------------------------------------------
# 3.3 Near-Duplicate Detection and Removal
# Purpose: Validate the cleaned corpus to ensure no invalid placeholders (e.g., 'No_Text') remain.
# Reasoning/Justification: Ensures the quality of the text data before further processing.
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




# Apply sentence tokenization
tokenised_corpus = cleaned_dataset.copy()

# ------------------------------------------------------------
# 4.2 Word Tokenisation
# Purpose: Split sentences into individual words or tokens for further processing.
# Reasoning/Justification: Provides the foundational structure for text vectorisation and similarity calculations.
# Notes:
# - Retains numerics and tokens relevant to the domain.
# ------------------------------------------------------------

# Word Tokenization Function
def word_tokenize_column(dataset, column):
    """Tokenize text into words and ensure the output is a Python list."""
    return dataset[column].apply(lambda x: word_tokenize(x) if pd.notna(x) else [])

# Apply Word Tokenization
tokenised_corpus['Words_Description'] = word_tokenize_column(tokenised_corpus, 'Consolidated_Description')
tokenised_corpus['Words_Subject_Matter'] = word_tokenize_column(tokenised_corpus, 'Consolidated_Subject_Matter')


# Save the tokenized dataset
print_dataset_summary(tokenised_corpus)
tokenised_corpus.to_csv(tokenised_corpus_path, index=False)
print("Tokenised Corpus saved successfully.")

# ------------------------------------------------------------
# 4.3 Bigram/Trigram Extraction
# Purpose: Identify multi-word phrases to enhance contextual representation.
# Reasoning/Justification: Captures semantic meaning by identifying commonly co-occurring terms (e.g., "data science").
# Notes:
# - Use statistical measures like PMI to filter meaningful bigrams/trigrams.
# ------------------------------------------------------------

def parse_stringified_lists(dataset, columns):
    """Safely parse stringified lists into Python lists."""
    for col in columns:
        dataset[col] = dataset[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return dataset

# Columns to parse
token_columns = [ 'Words_Description', 'Words_Subject_Matter']

# Parse the columns safely
tokenised_corpus = parse_stringified_lists(tokenised_corpus, token_columns)
logger.info("Stringified tokenized columns parsed into Python lists successfully.")

def extract_ngrams_for_document(tokens, ngram_range=(2, 3), min_df=1):
    try:
        if not tokens or len(tokens) < 2:
            return []  # Skip empty or short token lists
        text = " ".join(tokens)  # Join tokens into a single string
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words='english')
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()
    except ValueError:
        return []

# Apply bigram/trigram extraction
tokenised_corpus['Bigrams_Trigrams_Words_Description'] = tokenised_corpus['Words_Description'].apply(
    lambda x: extract_ngrams_for_document(x) if isinstance(x, list) else [])
tokenised_corpus['Bigrams_Trigrams_Words_Subject_Matter'] = tokenised_corpus['Words_Subject_Matter'].apply(
    lambda x: extract_ngrams_for_document(x) if isinstance(x, list) else [])


logger.info('Bigrams and Trigrams extracted successfully.')


# ------------------------------------------------------------
# 4.4 Named Entity Recognition (NER)
# Purpose: Identify named entities (e.g., person, organisation, location) in the text.
# Reasoning/Justification: Provides additional contextual features for downstream tasks.
# Notes:
# - Use pre-trained models like spaCy for NER.
# - Entities can include educational terms, locations, and institutions.
# ------------------------------------------------------------

def extract_named_entities(text, exclude_types={'CARDINAL', 'QUANTITY'}):
    """Extract named entities from text using spaCy, excluding specified entity types."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ not in exclude_types]

# Apply named entity recognition with filtering
tokenised_corpus['NER_Description'] = tokenised_corpus['Consolidated_Description'].apply(
    lambda x: extract_named_entities(x) if pd.notna(x) else [])
tokenised_corpus['NER_Subject_Matter'] = tokenised_corpus['Consolidated_Subject_Matter'].apply(
    lambda x: extract_named_entities(x) if pd.notna(x) else [])
tokenised_corpus['NER_Publisher'] = tokenised_corpus['Consolidated_Publisher'].apply(
    lambda x: extract_named_entities(x) if pd.notna(x) else [])
tokenised_corpus['NER_Author'] = tokenised_corpus['Consolidated_Author'].apply(
    lambda x: extract_named_entities(x) if pd.notna(x) else [])

logger.info("Named Entity Recognition completed with irrelevant entities excluded.")

# Save the updated tokenized dataset
tokenised_corpus.to_csv(tokenised_corpus_path, index=False)
logger.info("Tokenised Corpus with NER and n-grams saved successfully.")


# ------------------------------------------------------------
# 4.5 Corpus Validation
# Purpose: Validate the cleaned corpus to ensure no invalid placeholders (e.g., 'No_Text') remain.
# Reasoning/Justification: Ensures the quality of the text data before further processing.
# ------------------------------------------------------------

# Drop unused features Consolidated_Description	Consolidated_Subject_Matter	Consolidated_Publisher	Consolidated_Author	Consolidated_Title
tokenised_corpus = tokenised_corpus.drop(columns=['Consolidated_Description', 'Consolidated_Subject_Matter', 'Consolidated_Publisher', 'Consolidated_Author', 'Consolidated_Title'])


# Save the tokenized dataset with NER and n-grams
print_dataset_summary(tokenised_corpus)
tokenised_corpus.to_csv(tokenised_corpus_path, index=False)
print("Tokenised Corpus with NER and n-grams saved successfully.")


# ************************************************************
# ============================================================
# MILESTONE 2. PRODUCE A PRE-PROCESSED TEXT CORPUS
#
# - Normalised Text Data
# - Tokenised Text Data
# - Bigram/Trigram Extraction
# - Named Entity Recognition
#
# ============================================================
# ************************************************************



# ============================================================
# 5. BUILDING THE DOCUMENT MATRIX
# ============================================================



# ------------------------------------------------------------
# 5.1 Create Recommendation Profiles for School, Year Group and Subject.
# Purpose: Generate user profiles based on school, year group, and subject preferences.
# Reasoning/Justification: Profiles enable personalised recommendations by capturing user preferences.
# Notes:
# - 
# - 
# ------------------------------------------------------------




# ------------------------------------------------------------
# 5.2 TF-IDF Vectorisation
# Purpose: Convert the tokenised text data into a Document-Term Matrix using TF-IDF.
# Reasoning/Justification: TF-IDF helps in representing the importance of terms in documents.
# Notes:
# - Use TF-IDF vectorizer from sklearn.
# - Apply vectorization on the 'Words_Description', 'Words_Subject_Matter', 'Bigrams_Trigrams_Words_Description', 'Bigrams_Trigrams_Words_Subject_Matter', 'NER_Description', 'NER_Subject_Matter', and 'NER_Publisher' columns.
# ------------------------------------------------------------



def create_tfidf_matrix(corpus, max_features=5000):
    """Create a TF-IDF matrix for the given corpus."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

# Convert token lists back to strings for TF-IDF vectorization
def join_tokens(tokens):
    return ' '.join([token if isinstance(token, str) else token[0] for token in tokens])

columns_to_vectorize = [
    'Words_Description', 'Words_Subject_Matter',
    'Bigrams_Trigrams_Words_Description', 'Bigrams_Trigrams_Words_Subject_Matter',
    'NER_Description', 'NER_Subject_Matter', 'NER_Publisher'
]

for column in columns_to_vectorize:
    tokenised_corpus[column + '_Text'] = Parallel(n_jobs=-1)(delayed(join_tokens)(tokens) for tokens in tokenised_corpus[column])

# Create TF-IDF matrices for each feature
tfidf_matrices = {}
tfidf_vectorizers = {}

for column in columns_to_vectorize:
    tfidf_matrices[column], tfidf_vectorizers[column] = create_tfidf_matrix(tokenised_corpus[column + '_Text'])

# Save the TF-IDF matrices and vectorizers for later use
import joblib

for column in columns_to_vectorize:
    matrix_path = f'c:\\university\\A1-NLP-Recommendation-Engine\\Outputs\\models\\tfidf_matrix_{column.lower()}.pkl'
    vectorizer_path = f'c:\\university\\A1-NLP-Recommendation-Engine\\Outputs\\models\\tfidf_vectorizer_{column.lower()}.pkl'
    joblib.dump(tfidf_matrices[column], matrix_path)
    joblib.dump(tfidf_vectorizers[column], vectorizer_path)

logger.info("TF-IDF matrices and vectorizers saved successfully.")

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




# ************************************************************
# ============================================================
# MILESTONE 3. PRODUCE A DOCUMENT-TERM MATRIX
#
# Deliverables:
# - Generate and save a TF-IDF document-term matrix for the entire corpus.
# - Create and save individual TF-IDF matrices for key features: descriptions, subject matter, and NER outputs.
# - Ensure inclusion of additional features such as bigrams and trigrams for enriched representation.
# - Integrate and normalize profile data (School, Subject, Year Group) into the document-term matrix.
# - Inspect and document sparsity levels in the generated matrices.
# - Save all processed matrices and relevant metadata for reproducibility.
#
# Steps:
# 1. Preprocess tokenized text and concatenate features into combined strings suitable for TF-IDF processing.
# 2. Generate TF-IDF matrices for descriptions, subject matter, and other key textual components.
# 3. Integrate profile data by appending normalized counts (School Sum, Subject_Sum, Year Group Sum) to the document-term matrix.
# 4. Evaluate the sparsity of the TF-IDF matrices to determine if dimensionality reduction is necessary.
# 5. If needed, apply dimensionality reduction techniques such as PCA or SVD to optimize efficiency.
# 6. Save the document-term matrix and individual matrices for downstream use in similarity and recommendation tasks.
# ============================================================
# ************************************************************












# ************************************************************
# ============================================================
# MILESTONE 4. PRODUCE A SIMILARITY MATRIX
#
# Deliverables:
# - Generate a cosine similarity matrix for the entire dataset.
# - Incorporate profile features (School, Subject, Year Group) into similarity calculations.
# - Produce visualizations (e.g., heatmaps) of the similarity matrix to identify patterns and clusters.
# - Save the similarity matrix for use in downstream tasks.
#
# Steps:
# 1. Combine all relevant features (e.g., descriptions, NER, bigrams/trigrams) and profiles into a single representation.
# 2. Generate a TF-IDF matrix from the combined features.
# 3. Compute the cosine similarity matrix using the TF-IDF vectors.
# 4. Visualize the cosine similarity matrix to identify potential clusters or patterns.
# 5. Save the cosine similarity matrix and related artifacts for reproducibility.
# ============================================================
# ************************************************************

# ------------------------------------------------------------
# 4.1 Combine Features for Similarity Calculation
# Purpose: Combine all relevant features into a single representation for similarity calculations.
# Reasoning/Justification: Using a comprehensive set of features improves the accuracy of the similarity calculations.
# ------------------------------------------------------------



def combine_features(row):
    """Combine all relevant features into a single string."""
    return ' '.join(row['Words_Description']) + ' ' + \
           ' '.join(row['Words_Subject_Matter']) + ' ' + \
           ' '.join(row['Bigrams_Trigrams_Words_Description']) + ' ' + \
           ' '.join(row['Bigrams_Trigrams_Words_Subject_Matter']) + ' ' + \
           ' '.join([ent[0] for ent in row['NER_Description']]) + ' ' + \
           ' '.join([ent[0] for ent in row['NER_Subject_Matter']]) + ' ' + \
           ' '.join([ent[0] for ent in row['NER_Publisher']])

# Apply the function to combine features using parallel processing
tokenised_corpus['Combined_Features'] = Parallel(n_jobs=-1)(delayed(combine_features)(row) for _, row in tokenised_corpus.iterrows())

# ------------------------------------------------------------
# 4.2 Calculate Cosine Similarity
# Purpose: Calculate the cosine similarity between documents based on the combined features.
# Reasoning/Justification: Cosine similarity is a common measure for text similarity, capturing the angle between vectors.
# ------------------------------------------------------------

# Create TF-IDF matrix for combined features
tfidf_matrix_combined, tfidf_vectorizer_combined = create_tfidf_matrix(tokenised_corpus['Combined_Features'])

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix_combined, tfidf_matrix_combined)

# Save the cosine similarity matrix for later use
cosine_sim_matrix_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\Outputs\\models\\cosine_sim_matrix.pkl'
joblib.dump(cosine_sim_matrix, cosine_sim_matrix_path)

logger.info("Cosine similarity matrix saved successfully.")

# ------------------------------------------------------------
# 4.3 Visualise Cosine Similarity
# Purpose: Visualise the cosine similarity matrix to identify patterns and clusters.
# Reasoning/Justification: Visualisation aids in understanding the relationships between documents.
# ------------------------------------------------------------

# Plot the cosine similarity matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_sim_matrix, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity Matrix")
plt.show()





# ************************************************************
# ============================================================
# MILESTONE 5. PRODUCE A FUNCTIONAL RECOMMENDATION ENGINE
#
# - Filtering and Ranking
# - User Inputs and Profile Matching
# - 
# - 
#
# ============================================================
# ************************************************************



# ************************************************************
# ============================================================
# MILESTONE 6. VALIDATE AND OPTIMISE THE RECOMMENDATION ENGINE
#
# - Validation Metrics
# - Optimisation Techniques
# - Weighting and Scoring
# - Dimensionality Reduction
#
# ============================================================
# ************************************************************



# ************************************************************
# ============================================================
# MILESTONE 7. SUCCESSFUL DEMONSTRATION OF THE RECOMMENDATION ENGINE
#
# - Query Scenario 'A new school is looking for Science textbooks for year 9.'
# - 
# - 
# - 
#
# ============================================================
# ************************************************************





















