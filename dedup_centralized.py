# dedup_maintenance.py
import pandas as pd
from rapidfuzz import process, fuzz
from typing import Optional

def find_duplicates_in_main_list(
    main_file_path: str,
    vendor_name_column: str,
    code_column_name: str,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Analyzes a single list to find potential duplicates within it using fuzzy matching.

    Args:
        main_file_path (str): Path to the main list CSV to be checked.
        vendor_name_column (str): The name of the column containing vendor names.
        code_column_name (str): The name of the column containing the unique vendor code.
        similarity_threshold (int): The score (0-100) to consider a match a duplicate.

    Returns:
        pd.DataFrame: A DataFrame containing only the rows identified as potential duplicates,
                      with added columns for context.
    """
    try:
        main_df = pd.read_csv(main_file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the file {e.filename}.")
        return pd.DataFrame()

    # --- Robustness Checks ---
    if vendor_name_column not in main_df.columns or code_column_name not in main_df.columns:
        print(f"❌ Error: A required column was not found in '{main_file_path}'.")
        return pd.DataFrame()

    # --- Fuzzy Matching Logic ---
    choices = main_df[vendor_name_column].dropna().tolist()

    potential_duplicates = []

    # Iterate through each row of the DataFrame
    for index, row in main_df.iterrows():
        current_name = row[vendor_name_column]

        # Find the best match for the current name from the entire list
        # We need to find matches other than the item itself
        best_matches = process.extract(
            current_name,
            choices,
            scorer=fuzz.WRatio,
            limit=2 # Get the top 2 matches
        )

        # best_matches[0] will always be the item itself (score 100)
        # We check if there is a second-best match (best_matches[1])
        if len(best_matches) > 1 and best_matches[1][1] >= similarity_threshold:
            # A potential duplicate was found!
            matched_name, score, matched_index = best_matches[1]

            # Create a dictionary with info from the original row
            duplicate_info = row.to_dict()

            # Add the new remark columns
            duplicate_info['Remark'] = "Potential Duplicate"
            duplicate_info['Match_Score'] = int(score)
            duplicate_info['Duplicate_Of_Vendor'] = matched_name
            duplicate_info['Duplicate_Vendor_Code'] = main_df.iloc[matched_index][code_column_name]

            potential_duplicates.append(duplicate_info)

    if not potential_duplicates:
        print("✅ No potential duplicates found in the main list.")
        return pd.DataFrame()

    # Create a new DataFrame from the list of dictionaries
    duplicates_df = pd.DataFrame(potential_duplicates)
    return duplicates_df

# --- How to use the maintenance function ---
if __name__ == '__main__':
    # Assume main_list.csv has columns 'Vendor Name' and 'New Vendor Code'
    duplicate_report_df = find_duplicates_in_main_list(
        main_file_path='main_list.csv',
        vendor_name_column='Vendor Name',
        code_column_name='New Vendor Code', # The code column in your main list
        similarity_threshold=90
    )

    if not duplicate_report_df.empty:
        print(f"✅ Processing Complete. Found {len(duplicate_report_df)} potential duplicate entries.")

        output_filename = 'main_list_duplicate_report.csv'

        # Reorder columns for clarity in the report
        column_order = ['Remark', 'Match_Score', 'Duplicate_Of_Vendor', 'Duplicate_Vendor_Code'] + [col for col in duplicate_report_df.columns if col not in ['Remark', 'Match_Score', 'Duplicate_Of_Vendor', 'Duplicate_Vendor_Code']]
        duplicate_report_df = duplicate_report_df[column_order]

        duplicate_report_df.to_csv(output_filename, index=False)
        print(f"\n💾 Duplicate report saved to '{output_filename}'")
