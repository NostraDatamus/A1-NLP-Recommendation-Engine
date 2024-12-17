"""
===============================================================
Script Name : Data_Enrichment.py
Project Name: Assessment 1: NLP Recommendation Engine
Unit Name   : MA5851
Description : A script to enrich the dataset with additional metadata from Google Books, Trove, and OpenLibrary APIs.
Author      : Alexander Floyd
Student ID  : JC502993
Email       : alex.floyd@my.jcu.edu.au
Date        : November 30, 2024
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
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor

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

# Define file paths
spreadsheet_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\MA5851_A1_Data.xlsx'
output_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\enriched_dataset.xlsx'
consolidated_output_path = 'c:\\university\\A1-NLP-Recommendation-Engine\\datasets\\original_dataset.xlsx'

# Define ISBN List
isbn_data = pd.ExcelFile(spreadsheet_path).parse('ISBN_Data')

# API Keys
trove_api_key = "YGO9sqhZ0RgJpm2caSMaA59Bs13YHWo9"
google_api_key = "AIzaSyDACu3RqqmBGqF9zqqVNcbQNpQbI07Qv8c"

# ------------------------------------------------------------
# 1.4 Load Data
# Purpose: Load the enriched dataset for further processing.
# Reasoning/Justification: Ensures the dataset is available for subsequent steps.
# ------------------------------------------------------------
logging.info("Loading the original dataset...")
original_dataset = pd.read_excel(spreadsheet_path)
logging.info(f"Initial columns: {original_dataset.columns.tolist()}")


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 2.1 Category Consolidation
# Purpose: Consolidate multiple related categories into broader categories.
# Reasoning/Justification: Simplifies analysis and reporting by reducing the number of distinct categories.
# ------------------------------------------------------------
# Subject Field Consolidation Function
def subject_consolidation(dataset):
    """ Preprocess the dataset by adding an index for duplicate ISBNs and consolidating the 'Subject' field. """
    # Define the subject replacement rules
    subject_replacements = {
        # English consolidations
        'english': 'ENGLISH',
        'ENGLISH LITERARY STUDIES': 'ENGLISH',

        # Performance Arts consolidations
        'MUSIC': 'PERFORMANCE ARTS',
        'VISUAL ARTS': 'PERFORMANCE ARTS',
        'MEDIA': 'PERFORMANCE ARTS',
        'DRAMA': 'PERFORMANCE ARTS',
        'DANCE': 'PERFORMANCE ARTS',

        # Practical Studies consolidations
        'FOOD TECHNOLOGY': 'PRACTICAL STUDIES',
        'HOME ECONOMICS': 'PRACTICAL STUDIES',
        'WORK STUDIES': 'PRACTICAL STUDIES',
        'PDHPE': 'PRACTICAL STUDIES',

        # Business and Law consolidations
        'BUSINESS STUDIES': 'BUSINESS AND LAW',
        'ECONOMICS': 'BUSINESS AND LAW',
        'ACCOUNTING': 'BUSINESS AND LAW',
        'POLITICS AND LAW': 'BUSINESS AND LAW',
        'LEGAL STUDIES': 'BUSINESS AND LAW',

        # Design and Technology consolidations
        'COMPUTER SCIENCE': 'DESIGN AND TECHNOLOGY',
        'ENGINEERING STUDIES': 'DESIGN AND TECHNOLOGY',

        # Environmental Science consolidations
        'AGRICULTURE': 'ENVIRONMENTAL SCIENCE',
        'GEOGRAPHY': 'ENVIRONMENTAL SCIENCE',

        # Humanities consolidations
        'PHILOSOPHY': 'HUMANITIES',
        'PSYCHOLOGY': 'HUMANITIES',
        'RELIGIOUS EDUCATION': 'HUMANITIES',
    }

    # Make a copy of the dataset to avoid modifying the original
    processed_dataset = dataset.copy()

    # Add an index per ISBN grouping to distinguish duplicates
    #processed_dataset['idx'] = processed_dataset.groupby('ISBN').cumcount() + 1

    # Apply the replacement rules to consolidate 'Subject' field values
    processed_dataset['Subject'] = processed_dataset['Subject'].replace(subject_replacements)

    return processed_dataset

# ------------------------------------------------------------
# 2.2 Extract Unique ISBNs with Most Frequent Subject
# Purpose: Extract a list of unique ISBNs along with their most frequent subject.
# Reasoning/Justification: Ensures each ISBN is associated with the most relevant subject for further analysis.
# ------------------------------------------------------------
# Isolate Appropriate 'Subject' and Create Unique ISBN List Function
def get_unique_isbn_with_subject(dataset):
    """Get a dataframe with unique ISBNs and the most frequent 'Subject'."""
    
    # Group by ISBN and Subject to get the count of each subject per ISBN
    subject_counts = dataset.groupby(['ISBN', 'Subject']).size().reset_index(name='Subject_Count')
    
    # Get the most frequent subject for each ISBN
    idx = subject_counts.groupby('ISBN')['Subject_Count'].idxmax()
    most_frequent_subjects = subject_counts.loc[idx].reset_index(drop=True)
    
    # Merge with the original dataset to get the 'State' counts
    state_counts = dataset.groupby(['ISBN', 'Subject', 'State']).size().reset_index(name='State_Count')
    merged_counts = pd.merge(most_frequent_subjects, state_counts, on=['ISBN', 'Subject'], how='left')
    
    # Resolve ties by 'State' count
    def resolve_ties(group):
        if len(group) > 1:
            return group.loc[group['State_Count'].idxmax()]
        return group.iloc[0]
    
    resolved_subjects = merged_counts.groupby('ISBN').apply(resolve_ties).reset_index(drop=True)
    
    # Select the relevant columns
    unique_isbn_subjects = resolved_subjects[['ISBN', 'Subject']]
    
    return unique_isbn_subjects

# ------------------------------------------------------------
# 2.3 Create Recommendation Profiles
# Purpose: Unpivot the dataset to create profiles for schools, subjects, and year groups.
# Reasoning/Justification: Enables the generation of recommendations based on various metadata attributes.
# ------------------------------------------------------------
# Recommendation Profile Creation and Merge Function
def create_combined_profiles(dataframe):
    """Creates and combines School Profile, Subject Profile, and Year Group Profile from a given dataframe."""
    
    # School Profile
    school_profile = dataframe.pivot_table(index='ISBN', 
                                           columns='School_ID', 
                                           aggfunc='size', 
                                           fill_value=0)
    school_profile['School Sum'] = school_profile.sum(axis=1)
    school_profile.columns = [f"School {col}" if isinstance(col, int) else col for col in school_profile.columns]

    # Subject Profile
    subject_profile = dataframe.pivot_table(index='ISBN', 
                                            columns='Subject', 
                                            aggfunc='size', 
                                            fill_value=0)
    subject_profile['Subject_Sum'] = subject_profile.sum(axis=1)

    # Year Group Profile
    year_group_profile = dataframe.pivot_table(index='ISBN', 
                                               columns='Year', 
                                               aggfunc='size', 
                                               fill_value=0)
    year_group_profile['Year Group Sum'] = year_group_profile.sum(axis=1)
    year_group_profile.columns = [f"Year {col}" if isinstance(col, int) else col for col in year_group_profile.columns]

    # Combine the profiles
    combined_profiles = school_profile.merge(subject_profile, on='ISBN', how='outer').merge(year_group_profile, on='ISBN', how='outer')
    combined_profiles.reset_index(inplace=True)
    
    return combined_profiles

# ============================================================
# 2. DATA ENRICHMENT
# ============================================================

# ------------------------------------------------------------
# 2.1 Google Books API Enrichment
# Purpose: Retrieve additional metadata from the Google Books API.
# Reasoning/Justification: Enhances the dataset with more detailed information for better recommendations.
# ------------------------------------------------------------
# Google Books API Enrichment Function
def enrich_with_google_books(dataset):
    """Enrich the dataset by fetching book details from the Google Books API for each ISBN."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Function to fetch Google Books data for a single ISBN
    def fetch_google_books_data(isbn):
        #url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={google_api_key}"
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.warning(f"Failed to fetch data from Google Books for ISBN {isbn}. Status code: {response.status_code}")
                return None

            data = response.json()
            if "items" in data:
                volume_info = data["items"][0]["volumeInfo"]

                return {
                    'G_Title': volume_info.get('title', 'N/A'),
                    'G_Subtitle': volume_info.get('subtitle', 'N/A'),
                    'G_Author': ", ".join(volume_info.get('authors', [])) if 'authors' in volume_info else 'N/A',
                    'G_Publisher': volume_info.get('publisher', 'N/A'),
                    'G_Date': volume_info.get('publishedDate', 'N/A'),
                    'G_Description': volume_info.get('description', 'N/A'),
                    'G_Category': ", ".join(volume_info.get('categories', [])) if 'categories' in volume_info else 'N/A',
                    'G_Page_Count': volume_info.get('pageCount', 'N/A'),
                    'G_TextSnippet': data["items"][0].get("searchInfo", {}).get("textSnippet", 'N/A')
                }
            else:
                logging.info(f"No data found in Google Books for ISBN {isbn}.")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from Google Books for ISBN {isbn}: {e}")
            return None

    # Fetch data for each ISBN in the dataset
    google_books_data = []
    for isbn in dataset['ISBN'].unique():
        book_data = fetch_google_books_data(isbn)
        google_books_data.append(book_data if book_data else {
            'G_Title': 'N/A', 'G_Subtitle': 'N/A', 'G_Author': 'N/A', 'G_Publisher': 'N/A',
            'G_Date': 'N/A', 'G_Description': 'N/A', 'G_Category': 'N/A', 'G_Page_Count': 'N/A',
            'G_TextSnippet': 'N/A'
        })

    # Convert Google Books data into a DataFrame
    google_books_df = pd.DataFrame(google_books_data, index=dataset['ISBN'].unique())
    google_books_df.index.name = 'ISBN'
    google_books_df.reset_index(inplace=True)

    # Merge the new data with the original dataset
    enriched_dataset = dataset.merge(google_books_df, how='outer', on='ISBN')
    return enriched_dataset

# ------------------------------------------------------------
# 2.2 The Trove API Enrichment
# Purpose: Retrieve additional metadata from The Trove API.
# Reasoning/Justification: Enhances the dataset with more detailed information for better recommendations.
# ------------------------------------------------------------
def enrich_with_trove(dataset, trove_api_key):
    """Enrich the dataset by fetching book details from the Trove API for each ISBN."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Function to fetch Trove data for a single ISBN
    def fetch_trove_data(isbn):
        # Updated endpoint with version 3 and proper category parameter
        url = f"https://api.trove.nla.gov.au/v3/result?q=isbn%3A{isbn}&category=book&key={trove_api_key}&reclevel=full&include=tags&include=comments&encoding=json"
        try:
            response = requests.get(url)
            # Log and handle unsuccessful requests
            if response.status_code != 200:
                if response.status_code == 429:
                    logging.warning(f"Trove rate limit exceeded. Skipping ISBN {isbn}.")
                else:
                    logging.warning(f"Failed to fetch data from Trove for ISBN {isbn}. Status code: {response.status_code}")
                return None

            # Parse the JSON response
            data = response.json()
            category_data = data.get('category', [])
            if category_data and 'records' in category_data[0] and 'work' in category_data[0]['records']:
                works = category_data[0]['records']['work']
                work = works[0] if isinstance(works, list) else works

                # Extract fields
                tags = ", ".join(tag.get('value', 'N/A') for tag in work.get('tag', [])) if 'tag' in work else 'N/A'
                comments = ", ".join(comment.get('value', 'N/A') for comment in work.get('comment', [])) if 'comment' in work else 'N/A'
                is_part_of = ", ".join(part.get('value', 'N/A') for part in work.get('isPartOf', [])) if 'isPartOf' in work else 'N/A'

                return {
                    'T_Title': work.get('title', 'N/A'),
                    'T_Author': ", ".join(work.get('contributor', [])) if 'contributor' in work else 'N/A',
                    'T_Publisher': work.get("version", [{}])[0].get("publisher", 'N/A') if work.get("version") else 'N/A',
                    'T_Date': work.get('issued', 'N/A'),
                    'T_Description': work.get('description', 'N/A'),
                    'T_Tags': tags,
                    'T_Comments': comments,
                    'T_Part Of': is_part_of,
                    'T_Summary': work.get('summary', 'N/A'),
                    'T_Snippet': work.get('snippet', 'N/A')
                }
            else:
                logging.info(f"No data found in Trove for ISBN {isbn}.")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from Trove for ISBN {isbn}: {e}")
            return None

    # Fetch data for each ISBN in the dataset
    trove_data = []
    for isbn in dataset['ISBN'].unique():
        book_data = fetch_trove_data(isbn)
        trove_data.append(book_data if book_data else {
            'T_Title': 'N/A', 'T_Author': 'N/A', 'T_Publisher': 'N/A', 'T_Date': 'N/A',
            'T_Description': 'N/A', 'T_Tags': 'N/A', 'T_Comments': 'N/A',
            'T_Part Of': 'N/A', 'T_Summary': 'N/A', 'T_Snippet': 'N/A'
        })

    # Convert Trove data into a DataFrame
    trove_df = pd.DataFrame(trove_data, index=dataset['ISBN'].unique())
    trove_df.index.name = 'ISBN'
    trove_df.reset_index(inplace=True)

    # Merge the new data with the original dataset
    enriched_dataset = dataset.merge(trove_df, how='outer', on='ISBN')
    return enriched_dataset

# ------------------------------------------------------------
# 2.3 OpenLibrary API Enrichment
# Purpose: Retrieve additional metadata from The OpenLibrary API.
# Reasoning/Justification: Enhances the dataset with more detailed information for better recommendations.
# ------------------------------------------------------------
def enrich_with_openlibrary(dataset):
    """Enrich the dataset by fetching book details from the OpenLibrary API for each ISBN."""

    def fetch_openlibrary_data(isbn):
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.warning(f"Failed to fetch data from OpenLibrary for ISBN {isbn}. Status code: {response.status_code}")
                return None

            data = response.json()
            book_key = f"ISBN:{isbn}"
            if book_key in data:
                book_data = data[book_key]
                description = book_data.get('description', 'N/A')
                if isinstance(description, dict):  # Handle nested description
                    description = description.get('value', 'N/A')

                return {
                    'O_Title': book_data.get('title', 'N/A'),
                    'O_Subtitle': book_data.get('subtitle', 'N/A'),
                    'O_Author': ", ".join(author.get('name', 'N/A') for author in book_data.get('authors', [])) if 'authors' in book_data else 'N/A',
                    'O_Publisher': ", ".join(publisher.get('name', 'N/A') for publisher in book_data.get('publishers', [])) if 'publishers' in book_data else 'N/A',
                    'O_Date': book_data.get('publish_date', 'N/A'),
                    'O_Description': description,
                    'O_Excerpts': ", ".join(f"{excerpt.get('comment', 'N/A')} - {excerpt.get('text', 'N/A')}" for excerpt in book_data.get('excerpts', [])) if 'excerpts' in book_data else 'N/A',
                    'O_Subjects': ", ".join(subject.get('name', 'N/A') for subject in book_data.get('subjects', [])) if 'subjects' in book_data else 'N/A',
                    'O_Page_Count': book_data.get('number_of_pages', 'N/A')
                }
            else:
                logging.info(f"No data found in OpenLibrary for ISBN {isbn}.")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from OpenLibrary for ISBN {isbn}: {e}")
            return None

    openlibrary_data = []
    for isbn in dataset['ISBN'].unique():
        book_data = fetch_openlibrary_data(isbn)
        openlibrary_data.append(book_data if book_data else {
            'O_Title': 'N/A', 'O_Subtitle': 'N/A', 'O_Author': 'N/A', 'O_Publisher': 'N/A',
            'O_Date': 'N/A', 'O_Description': 'N/A', 'O_Excerpts': 'N/A',
            'O_Subjects': 'N/A', 'O_Page_Count': 'N/A'
        })

    openlibrary_df = pd.DataFrame(openlibrary_data, index=dataset['ISBN'].unique())
    openlibrary_df.index.name = 'ISBN'
    openlibrary_df.reset_index(inplace=True)
    return openlibrary_df

# ------------------------------------------------------------
# 2.4 Consolidated API Data Enrichment Function
# Purpose: Combine the Google Books, Trove, and OpenLibrary API enrichments into a single function.
# Reasoning/Justification: Simplifies the enrichment process by consolidating multiple API calls into one.
# ------------------------------------------------------------
def enrich_dataset_with_apis(dataset, trove_api_key):
    """Enrich the dataset by fetching data from Google Books, Trove, and OpenLibrary APIs for each ISBN."""

    # Validate input
    if 'ISBN' not in dataset.columns:
        raise KeyError("The 'ISBN' column is missing in the dataset.")

    # Initialize tracking variables
    total_no_data, google_missing, trove_missing, openlibrary_missing = 0, 0, 0, 0
    enriched_data = []

    unique_isbns = dataset['ISBN'].unique()
    logging.info(f"Processing {len(unique_isbns)} unique ISBNs.")

    for isbn in unique_isbns:
        if pd.isna(isbn):
            logging.warning(f"Missing ISBN for row. Skipping.")
            continue

        logging.info(f"Processing ISBN: {isbn}")

        google_data, trove_data, openlibrary_data = {}, {}, {}

        # Enrich with Google Books API
        try:
            google_data = enrich_with_google_books(pd.DataFrame({'ISBN': [isbn]}))
            if google_data is None or google_data.empty:
                logging.warning(f"No data from Google Books for ISBN {isbn}.")
                google_missing += 1
                google_data = {
                    'G_Title': 'N/A', 'G_Subtitle': 'N/A', 'G_Author': 'N/A', 'G_Publisher': 'N/A',
                    'G_Date': 'N/A', 'G_Description': 'N/A', 'G_Category': 'N/A',
                    'G_Page_Count': 'N/A', 'G_TextSnippet': 'N/A'
                }
            else:
                google_data = google_data.iloc[0].to_dict()
        except Exception as e:
            logging.error(f"Error in Google Books enrichment for ISBN {isbn}: {e}")
            google_missing += 1
            google_data = {
                'G_Title': 'N/A', 'G_Subtitle': 'N/A', 'G_Author': 'N/A', 'G_Publisher': 'N/A',
                'G_Date': 'N/A', 'G_Description': 'N/A', 'G_Category': 'N/A',
                'G_Page_Count': 'N/A', 'G_TextSnippet': 'N/A'
            }

        # Enrich with Trove API
        try:
            trove_data = enrich_with_trove(pd.DataFrame({'ISBN': [isbn]}), trove_api_key)
            if trove_data is None or trove_data.empty:
                logging.warning(f"No data from Trove for ISBN {isbn}.")
                trove_missing += 1
                trove_data = {
                    'T_Title': 'N/A', 'T_Author': 'N/A', 'T_Publisher': 'N/A', 'T_Date': 'N/A',
                    'T_Description': 'N/A', 'T_Tags': 'N/A', 'T_Comments': 'N/A',
                    'T_Part Of': 'N/A', 'T_Summary': 'N/A', 'T_Snippet': 'N/A'
                }
            else:
                trove_data = trove_data.iloc[0].to_dict()
        except Exception as e:
            logging.error(f"Error in Trove enrichment for ISBN {isbn}: {e}")
            trove_missing += 1
            trove_data = {
                'T_Title': 'N/A', 'T_Author': 'N/A', 'T_Publisher': 'N/A', 'T_Date': 'N/A',
                'T_Description': 'N/A', 'T_Tags': 'N/A', 'T_Comments': 'N/A',
                'T_Part Of': 'N/A', 'T_Summary': 'N/A', 'T_Snippet': 'N/A'
            }

        # Enrich with OpenLibrary API
        try:
            openlibrary_data = enrich_with_openlibrary(pd.DataFrame({'ISBN': [isbn]}))
            if openlibrary_data is None or openlibrary_data.empty:
                logging.warning(f"No data from OpenLibrary for ISBN {isbn}.")
                openlibrary_missing += 1
                openlibrary_data = {
                    'O_Title': 'N/A', 'O_Subtitle': 'N/A', 'O_Author': 'N/A', 'O_Publisher': 'N/A',
                    'O_Date': 'N/A', 'O_Description': 'N/A', 'O_Excerpts': 'N/A',
                    'O_Subjects': 'N/A', 'O_Page_Count': 'N/A'
                }
            else:
                openlibrary_data = openlibrary_data.iloc[0].to_dict()
        except Exception as e:
            logging.error(f"Error in OpenLibrary enrichment for ISBN {isbn}: {e}")
            openlibrary_missing += 1
            openlibrary_data = {
                'O_Title': 'N/A', 'O_Subtitle': 'N/A', 'O_Author': 'N/A', 'O_Publisher': 'N/A',
                'O_Date': 'N/A', 'O_Description': 'N/A', 'O_Excerpts': 'N/A',
                'O_Subjects': 'N/A', 'O_Page_Count': 'N/A'
            }

        # Combine all data
        combined_data = {
            **google_data,
            **trove_data,
            **openlibrary_data,
            'ISBN': isbn  # Ensure ISBN is retained
        }

        # Check for missing data
        if combined_data.get('G_Title', 'N/A') == 'N/A':
            google_missing += 1
        if combined_data.get('T_Title', 'N/A') == 'N/A':
            trove_missing += 1
        if combined_data.get('O_Title', 'N/A') == 'N/A':
            openlibrary_missing += 1
        if all(value == 'N/A' for value in combined_data.values() if value != 'ISBN'):
            total_no_data += 1
            logging.warning(f"No data returned for ISBN {isbn}.")

        enriched_data.append(combined_data)
        time.sleep(0.1)  # Rate limiting

    # Convert to DataFrame
    enriched_df = pd.DataFrame(enriched_data).fillna('N/A')
    assert 'ISBN' in enriched_df.columns, "ISBN is missing from the enriched DataFrame."

    # Log summary
    logging.info(f"Processed {len(unique_isbns)} ISBNs. Total no data: {total_no_data}, "
                 f"Google missing: {google_missing}, Trove missing: {trove_missing}, "
                 f"OpenLibrary missing: {openlibrary_missing}.")

    # Merge with original dataset
    final_dataset = dataset.merge(enriched_df, how='outer', on='ISBN')

    return final_dataset


# ============================================================
# 3. Apply API Data Enrichment
# ============================================================

# ------------------------------------------------------------
# 3.1 Apply API Data Enrichment
# Purpose: Enrich the dataset with additional metadata from Google Books, Trove, and OpenLibrary APIs.
# Reasoning/Justification: Enhances the dataset with more detailed information for better recommendations.
# ------------------------------------------------------------

# Consolidate Subject Categories
logging.info("Consolidating subjects...")
consolidated_dataset = subject_consolidation(isbn_data)
consolidated_dataset.to_excel(consolidated_output_path, index=False)

# Create Profiles
logging.info("Creating and merging profiles...")
profiled_dataset = create_combined_profiles(consolidated_dataset)
print("Profiled Dataset")

# Extract ISBNs with the most frequent 'Subject'
unique_isbn_subjects = get_unique_isbn_with_subject(consolidated_dataset)
# Merge the most frequent subjects back into the consolidated dataset
profiled_dataset = pd.merge(profiled_dataset, unique_isbn_subjects, on='ISBN', how='left')

# Enrich Dataset with APIs
logging.info("Enriching dataset with APIs...")
enriched_dataset = enrich_dataset_with_apis(profiled_dataset, trove_api_key)
logging.info(f"Columns after API enrichment: {enriched_dataset.columns.tolist()}")

# ------------------------------------------------------------
# 3.2 Export Enriched Dataset
# Purpose: Save the enriched dataset to a new file.
# Reasoning/Justification: Ensures the enriched dataset is available for use in other scripts.
# ------------------------------------------------------------
enriched_dataset.to_excel(output_path, index=False)
logging.info("Data retrieval and saving complete.")