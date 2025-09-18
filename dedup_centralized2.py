# dedup_maintenance_v2.py
import pandas as pd
from rapidfuzz import process, fuzz
from typing import Optional
import re

def standardize_name(name: str) -> str:
    """
    Cleans and standardizes a vendor name for better fuzzy matching.
    - Converts to lowercase
    - Removes common corporate suffixes (inc, corp, llc, etc.)
    - Removes punctuation
    - Strips extra whitespace
    """
    if not isinstance(name, str):
        return ""

    name = name.lower()

    # Remove common suffixes using regular expressions for whole words
    suffixes = ['inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'llc', 'ltd']
    for suffix in suffixes:
        name = re.sub(r'\b' + suffix + r'\b\.?', '', name, flags=re.IGNORECASE) # \b is a word boundary

    # Remove all non-alphanumeric characters (keeps letters and numbers)
    name = re.sub(r'[^\w\s]', '', name)

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name.strip()


def find_duplicates_in_main_list(
    main_file_path: str,
    vendor_name_column: str,
    code_column_name: str,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Analyzes a single list to find potential duplicates within it using fuzzy matching.
    """
    try:
        main_df = pd.read_csv(main_file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the file {e.filename}.")
        return pd.DataFrame()

    if vendor_name_column not in main_df.columns or code_column_name not in main_df.columns:
        print(f"❌ Error: A required column was not found in '{main_file_path}'.")
        return pd.DataFrame()

    # --- NEW: Apply the standardization to a new column ---
    main_df['clean_name'] = main_df[vendor_name_column].apply(standardize_name)

    # --- Fuzzy Matching now uses the 'clean_name' column ---
    choices = main_df['clean_name'].dropna().tolist()

    potential_duplicates = []

    for index, row in main_df.iterrows():
        # Use the cleaned name for matching
        current_clean_name = row['clean_name']

        best_matches = process.extract(
            current_clean_name,
            choices,
            scorer=fuzz.WRatio,
            limit=2
        )

        if len(best_matches) > 1 and best_matches[1][1] >= similarity_threshold:
            matched_clean_name, score, matched_index = best_matches[1]

            # Ensure we're not just matching an empty string to another empty string
            if not current_clean_name or not matched_clean_name:
                continue

            duplicate_info = row.to_dict()

            # Use original names and codes for the report for better context
            duplicate_info['Remark'] = "Potential Duplicate"
            duplicate_info['Match_Score'] = int(score)
            duplicate_info['Duplicate_Of_Vendor'] = main_df.iloc[matched_index][vendor_name_column]
            duplicate_info['Duplicate_Vendor_Code'] = main_df.iloc[matched_index][code_column_name]

            potential_duplicates.append(duplicate_info)

    if not potential_duplicates:
        print("✅ No potential duplicates found in the main list.")
        return pd.DataFrame()

    duplicates_df = pd.DataFrame(potential_duplicates)
    # We can drop the temporary clean_name column from the final report
    duplicates_df = duplicates_df.drop(columns=['clean_name'], errors='ignore')
    return duplicates_df

# --- How to use the maintenance function ---
if __name__ == '__main__':
    duplicate_report_df = find_duplicates_in_main_list(
        main_file_path='main_list.csv',
        vendor_name_column='Vendor Name',
        code_column_name='New Vendor Code', # Assuming this is the correct code column
        similarity_threshold=90
    )

    if not duplicate_report_df.empty:
        print(f"✅ Processing Complete. Found {len(duplicate_report_df)} potential duplicate entries.")

        output_filename = 'main_list_duplicate_report_v2.csv'

        # Define column order for the report
        report_columns = [
            'Vendor Name', 'New Vendor Code', 'Remark', 'Match_Score',
            'Duplicate_Of_Vendor', 'Duplicate_Vendor_Code'
        ]
        # Get remaining columns from the original file
        other_columns = [col for col in duplicate_report_df.columns if col not in report_columns]
        final_columns = report_columns + other_columns

        duplicate_report_df = duplicate_report_df[final_columns]

        duplicate_report_df.to_csv(output_filename, index=False)
        print(f"\n💾 Duplicate report saved to '{output_filename}'")
