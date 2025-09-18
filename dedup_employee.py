# employee_dedup_library.py
import pandas as pd
from rapidfuzz import process, fuzz
from typing import Optional
import re

def standardize_person_name(name: str) -> str:
    """
    Parses and standardizes a person's name from 'LAST, FIRST M.' format
    into a clean 'firstname middlename lastname' format for better matching.

    Args:
        name (str): The name string, expected as 'LAST, FIRST M.'.

    Returns:
        str: A cleaned, standardized full name.
    """
    if not isinstance(name, str):
        return ""

    name = name.lower()

    # Handle suffixes like jr, sr, iii, etc.
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
    for suffix in suffixes:
        name = re.sub(r'\b' + suffix + r'\b\.?', '', name, flags=re.IGNORECASE)

    parts = [part.strip() for part in name.split(',')]

    # Rearrange from "last, first m" to "first m last"
    if len(parts) == 2:
        last_name = parts[0]
        first_middle_name = parts[1]
        standard_name = f"{first_middle_name} {last_name}"
    else:
        # If format is not as expected, just use the original name
        standard_name = name

    # Remove all punctuation and extra whitespace
    standard_name = re.sub(r'[^\w\s]', '', standard_name)
    standard_name = ' '.join(standard_name.split())

    return standard_name.strip()

def check_employee_duplicates(
    main_file_path: str,
    incoming_file_path: str,
    name_column: str,
    code_column: Optional[str] = None,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Compares an incoming list of employees against a main list using fuzzy matching
    on standardized person names.
    """
    try:
        main_df = pd.read_csv(main_file_path)
        incoming_df = pd.read_csv(incoming_file_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the file {e.filename}.")
        return pd.DataFrame()

    # --- Robustness Checks ---
    if name_column not in main_df.columns or name_column not in incoming_df.columns:
        print(f"❌ Error: Name column '{name_column}' not found in one of the files.")
        return pd.DataFrame()
    if code_column and code_column not in main_df.columns:
        print(f"❌ Error: Code column '{code_column}' not found in the main list.")
        return pd.DataFrame()

    # --- Apply the NEW person name standardization ---
    main_df['clean_name'] = main_df[name_column].apply(standardize_person_name)
    incoming_df['clean_name'] = incoming_df[name_column].apply(standardize_person_name)

    # --- Fuzzy Matching Logic (operates on the clean names) ---
    choices = main_df['clean_name'].dropna().tolist()
    remarks = []
    scores = []
    matched_codes = []
    matched_names = []

    for _, row in incoming_df.iterrows():
        current_clean_name = row['clean_name']
        if not current_clean_name:
            best_match = None
        else:
            best_match = process.extractOne(
                current_clean_name,
                choices,
                scorer=fuzz.WRatio,
                score_cutoff=similarity_threshold
            )

        if best_match:
            match_name, score, index = best_match
            remarks.append(f"Potential Duplicate")
            scores.append(int(score))

            # Retrieve original name and code from main list for the report
            original_matched_name = main_df.iloc[index][name_column]
            matched_names.append(original_matched_name)

            if code_column:
                matched_codes.append(main_df.iloc[index][code_column])
        else:
            remarks.append("New Employee")
            scores.append(0)
            matched_names.append("")
            if code_column:
                matched_codes.append("")

    # --- Build the final report ---
    result_df = incoming_df.copy()
    result_df['Remarks'] = remarks
    result_df['Match_Score'] = scores
    result_df['Matched_Name_In_Main_List'] = matched_names
    if code_column:
        result_df['Existing_Employee_Code'] = matched_codes

    # Drop the temporary clean name column
    result_df = result_df.drop(columns=['clean_name'])
    return result_df

# --- How to use the function for your files ---
if __name__ == '__main__':
    # Using the exact column names from your files
    processed_employees = check_employee_duplicates(
        main_file_path='main_list.csv',
        incoming_file_path='new_list.csv',
        name_column='Vendor Name',          # Column with employee names
        code_column='New Vendor Code',      # Column with employee codes
        similarity_threshold=90             # Threshold can be adjusted
    )

    if not processed_employees.empty:
        print("✅ Processing Complete. First 5 rows of output:")
        print(processed_employees.head())

        output_filename = 'employee_duplicate_report.csv'
        processed_employees.to_csv(output_filename, index=False)
        print(f"\n💾 Results saved to '{output_filename}'")
    else:
        print("No data processed or an error occurred.")
