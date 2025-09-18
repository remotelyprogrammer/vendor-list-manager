# dedup_library.py
import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Optional

def check_vendors_with_fuzzy_matching(
    main_file_path: str,
    incoming_file_path: str,
    vendor_name_column: str,
    code_column_name: Optional[str] = None,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Compares an incoming list of vendors against a main list using fuzzy matching.
    It can also retrieve the vendor code for matched duplicates.

    Args:
        main_file_path (str): Path to the main list CSV.
        incoming_file_path (str): Path to the incoming list CSV to be checked.
        vendor_name_column (str): The name of the column containing vendor names.
        code_column_name (Optional[str]): The name of the vendor code column in the main list.
        similarity_threshold (int): The score (0-100) to consider a match a duplicate.

    Returns:
        pd.DataFrame: The incoming DataFrame with added remarks and vendor code columns.
    """
    try:
        main_df = pd.read_csv(main_file_path)
        incoming_df = pd.read_csv(incoming_file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the file {e.filename}.")
        return pd.DataFrame()

    # --- Robustness Checks ---
    if vendor_name_column not in main_df.columns or vendor_name_column not in incoming_df.columns:
        print(f"❌ Error: Vendor name column '{vendor_name_column}' not found in one of the files.")
        return pd.DataFrame()
    if code_column_name and code_column_name not in main_df.columns:
        print(f"❌ Error: Vendor code column '{code_column_name}' not found in the main list.")
        return pd.DataFrame()

    # --- Fuzzy Matching Logic ---
    main_vendor_choices = main_df[vendor_name_column].dropna().tolist()
    remarks = []
    scores = []
    matched_codes = []

    for name in incoming_df[vendor_name_column]:
        best_match = process.extractOne(
            name,
            main_vendor_choices,
            scorer=fuzz.WRatio,
            score_cutoff=similarity_threshold
        )

        if best_match:
            # A match was found!
            # best_match is a tuple: (matched_name, score, index_in_main_list)
            matched_name, score, index = best_match

            remarks.append(f"Potential Duplicate of '{matched_name}'")
            scores.append(int(score))

            # Use the index to look up the vendor code if the column is provided
            if code_column_name:
                code = main_df.iloc[index][code_column_name]
                matched_codes.append(code)

        else:
            # No match found
            remarks.append("New Vendor")
            scores.append(0)
            if code_column_name:
                matched_codes.append("") # Append a blank if no match

    # Add the detailed remarks to the DataFrame
    result_df = incoming_df.copy()
    result_df['Remarks'] = remarks
    result_df['Match_Score'] = scores
    if code_column_name:
        result_df['Existing_Vendor_Code'] = matched_codes

    return result_df

# --- How to use the updated function ---
if __name__ == '__main__':
    # Assume main_list.csv has columns 'Vendor Name' and 'Vendor Code'
    processed_vendors = check_vendors_with_fuzzy_matching(
        main_file_path='main_list.csv',
        incoming_file_path='new_list.csv',
        vendor_name_column='Vendor Name',
        code_column_name='New Vendor Code', # <-- Specify your code column here
        similarity_threshold=90
    )

    if not processed_vendors.empty:
        print("✅ Processing Complete. First 5 rows of output:")
        print(processed_vendors.head())

        output_filename = 'new_list_with_codes_and_remarks.csv'
        processed_vendors.to_csv(output_filename, index=False)
        print(f"\n💾 Results saved to '{output_filename}'")
