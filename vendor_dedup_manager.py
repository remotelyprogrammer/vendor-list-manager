#!/usr/bin/env python3
# vendor_dedup_manager.py
"""
A unified solution for vendor and employee name deduplication in ERP systems.

This script provides a complete solution for:
1. Checking incoming vendor/employee names against a main list
2. Finding potential duplicates within a main list
3. Handling both vendor and employee name formats
4. Saving all output files to an 'output' directory

Author: GitHub Copilot
Created: September 2025
"""

import pandas as pd
import re
import argparse
import os
import sys
from rapidfuzz import process, fuzz
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


# --- Name Standardization Functions ---

def standardize_name(name: str, entity_type: str = "vendor") -> str:
    """
    Standardizes a name for better fuzzy matching based on the entity type.
    
    Args:
        name (str): The name string to standardize
        entity_type (str): Either "vendor" or "employee" to determine standardization approach
        
    Returns:
        str: A cleaned, standardized name
    """
    if not isinstance(name, str):
        return ""
    
    name = name.lower()
    
    if entity_type == "vendor":
        # Corporate name standardization for vendors
        suffixes = ['inc', 'incorporated', 'corp', 'corporation', 'co', 
                   'company', 'llc', 'ltd', 'limited', 'international']
        for suffix in suffixes:
            name = re.sub(r'\b' + suffix + r'\b\.?', '', name, flags=re.IGNORECASE)
    
    elif entity_type == "employee":
        # Handle person name format: "LAST, FIRST M."
        suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
        for suffix in suffixes:
            name = re.sub(r'\b' + suffix + r'\b\.?', '', name, flags=re.IGNORECASE)
            
        parts = [part.strip() for part in name.split(',')]
        
        # Rearrange from "last, first m" to "first m last"
        if len(parts) == 2:
            last_name = parts[0]
            first_middle_name = parts[1]
            name = f"{first_middle_name} {last_name}"
    
    # Common cleaning for both types
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
    name = ' '.join(name.split())         # Remove extra whitespace
    
    return name.strip()


# --- Core Comparison Functions ---

def find_duplicates_in_main_list(
    main_df: pd.DataFrame,
    name_column: str,
    code_column: str,
    entity_type: str = "vendor",
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Analyzes a single list to find potential duplicates within it using fuzzy matching.
    
    Args:
        main_df (pd.DataFrame): The main DataFrame to check for internal duplicates
        name_column (str): The column containing names to check
        code_column (str): The column containing unique codes
        entity_type (str): Either "vendor" or "employee" to determine standardization
        similarity_threshold (int): The score (0-100) to consider a match a duplicate
        
    Returns:
        pd.DataFrame: A DataFrame containing only potential duplicates with detailed match info
    """
    # Apply standardization to a new column
    main_df['clean_name'] = main_df[name_column].apply(
        lambda x: standardize_name(x, entity_type)
    )
    
    # Create list of standardized names for matching
    choices = main_df['clean_name'].dropna().tolist()
    
    potential_duplicates = []
    
    for index, row in main_df.iterrows():
        current_clean_name = row['clean_name']
        
        if not current_clean_name:
            continue
            
        # Find matches other than the item itself
        best_matches = process.extract(
            current_clean_name,
            choices,
            scorer=fuzz.WRatio,
            limit=2  # Get the top 2 matches (first will be itself)
        )
        
        # Check for a second-best match that meets the threshold
        if len(best_matches) > 1 and best_matches[1][1] >= similarity_threshold:
            matched_clean_name, score, matched_index = best_matches[1]
            
            # Ensure we're not matching empty strings
            if not current_clean_name or not matched_clean_name:
                continue
                
            # Create a duplicate info dictionary
            duplicate_info = row.to_dict()
            duplicate_info['Remark'] = "Potential Duplicate"
            duplicate_info['Match_Score'] = int(score)
            duplicate_info['Duplicate_Of_Entity'] = main_df.iloc[matched_index][name_column]
            duplicate_info['Duplicate_Entity_Code'] = main_df.iloc[matched_index][code_column]
            
            potential_duplicates.append(duplicate_info)
    
    if not potential_duplicates:
        print("✅ No potential duplicates found in the main list.")
        return pd.DataFrame()
        
    # Create DataFrame from results and drop the temporary column
    duplicates_df = pd.DataFrame(potential_duplicates)
    duplicates_df = duplicates_df.drop(columns=['clean_name'], errors='ignore')
    
    return duplicates_df


def check_incoming_against_main(
    main_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    name_column: str,
    code_column: Optional[str] = None,
    entity_type: str = "vendor",
    entity_type_column: Optional[str] = None,
    similarity_threshold: int = 90
) -> pd.DataFrame:
    """
    Compares an incoming list against a main list using fuzzy matching.
    
    Args:
        main_df (pd.DataFrame): The main reference DataFrame
        incoming_df (pd.DataFrame): The incoming DataFrame to check
        name_column (str): The column containing names to check
        code_column (Optional[str]): The column containing unique codes in main list
        entity_type (str): Default entity type ("vendor" or "employee") if no type column is provided
        entity_type_column (Optional[str]): The column in incoming_df that specifies entity type
        similarity_threshold (int): The score (0-100) to consider a match a duplicate
        
    Returns:
        pd.DataFrame: The incoming DataFrame with added remarks and match info
    """
    # Handle case where we have an entity type column in the incoming data
    if entity_type_column and entity_type_column in incoming_df.columns:
        print(f"✅ Using entity type column: '{entity_type_column}' for dynamic standardization")
        
        # Make a copy of main_df for each entity type we'll need
        main_df_vendor = main_df.copy()
        main_df_employee = main_df.copy()
        
        # Create standardized names for both entity types in main list
        main_df_vendor['clean_name'] = main_df_vendor[name_column].apply(
            lambda x: standardize_name(x, "vendor")
        )
        main_df_employee['clean_name'] = main_df_employee[name_column].apply(
            lambda x: standardize_name(x, "employee")
        )
        
        # Create clean names in incoming using the appropriate standardization per row
        def standardize_based_on_type(row):
            # Handle "Group" column values (both "Vendor" and "Employee" in any case)
            # Empty values or missing values default to vendor
            row_type = str(row.get(entity_type_column, "")).strip().lower()
            
            if row_type == "" or row_type is None:
                # Empty values default to vendor
                print(f"ℹ️ Empty group value for '{row.get(name_column, '')}' - treating as vendor")
                return standardize_name(row[name_column], "vendor")
            elif "employee" in row_type:
                return standardize_name(row[name_column], "employee")
            else:  # Default to vendor for anything else
                return standardize_name(row[name_column], "vendor")
            
        incoming_df['clean_name'] = incoming_df.apply(standardize_based_on_type, axis=1)
        
        # Store mapping of entity types for later use (will help with appropriate remarks)
        entity_types = {}
        for idx, row in incoming_df.iterrows():
            row_type = str(row.get(entity_type_column, "")).strip().lower()
            
            # Empty values default to vendor
            if row_type == "" or row_type is None:
                entity_types[idx] = "vendor"
            else:
                entity_types[idx] = "employee" if "employee" in row_type else "vendor"
            
    else:
        # Use the single entity_type parameter for all entries (original behavior)
        main_df['clean_name'] = main_df[name_column].apply(
            lambda x: standardize_name(x, entity_type)
        )
        incoming_df['clean_name'] = incoming_df[name_column].apply(
            lambda x: standardize_name(x, entity_type)
        )
        # All entries will use the same entity type
        entity_types = {idx: entity_type for idx in incoming_df.index}
    
    # Create list of standardized names from main list
    if entity_type_column and entity_type_column in incoming_df.columns:
        # For per-row type checking, we'll create separate choices for each type
        vendor_choices = main_df_vendor['clean_name'].dropna().tolist()
        employee_choices = main_df_employee['clean_name'].dropna().tolist() 
    else:
        # Single choice list for the standard case
        choices = main_df['clean_name'].dropna().tolist()
        
    # Prepare result columns
    remarks = []
    scores = []
    matched_codes = []
    matched_names = []
    
    for idx, row in incoming_df.iterrows():
        current_clean_name = row['clean_name']
        # Get the entity type for this specific row
        row_entity_type = entity_types[idx]
        entity_label = "Employee" if row_entity_type == "employee" else "Vendor"
        
        if not current_clean_name:
            best_match = None
        else:
            # Select the appropriate choices list based on entity type
            if entity_type_column and entity_type_column in incoming_df.columns:
                if row_entity_type == "employee":
                    current_choices = employee_choices
                    current_main_df = main_df_employee
                else:
                    current_choices = vendor_choices
                    current_main_df = main_df_vendor
            else:
                current_choices = choices
                current_main_df = main_df
            
            best_match = process.extractOne(
                current_clean_name,
                current_choices,
                scorer=fuzz.WRatio,
                score_cutoff=similarity_threshold
            )
        
        if best_match:
            match_name, score, index = best_match
            
            # Format remarks based on entity type
            if row_entity_type == "employee":
                remarks.append(f"Potential Duplicate {entity_label}")
            else:
                # For vendors, include the matched name as in your original code
                matched_vendor_name = current_main_df.iloc[index][name_column]
                remarks.append(f"Potential Duplicate of '{matched_vendor_name}'")
                
            scores.append(int(score))
            
            # Add original matched name from main list
            matched_names.append(current_main_df.iloc[index][name_column])
            
            # Add code if column specified
            if code_column:
                matched_codes.append(current_main_df.iloc[index][code_column])
        else:
            remarks.append(f"New {entity_label}")
            scores.append(0)
            matched_names.append("")
            if code_column:
                matched_codes.append("")
    
    # Build result DataFrame
    result_df = incoming_df.copy()
    result_df['Remarks'] = remarks
    result_df['Match_Score'] = scores
    
    # Use more generic column names when we have mixed types
    if entity_type_column and entity_type_column in incoming_df.columns:
        result_df['Matched_Name_In_Main_List'] = matched_names
        if code_column:
            result_df['Existing_Code'] = matched_codes
    else:
        # Use the original type-specific column names
        result_df[f'Matched_{entity_type.capitalize()}_In_Main'] = matched_names
        if code_column:
            result_df[f'Existing_{entity_type.capitalize()}_Code'] = matched_codes
    
    # Drop the temporary clean name column
    result_df = result_df.drop(columns=['clean_name'], errors='ignore')
    
    return result_df


# --- File Handling Functions ---

def load_dataframe(file_path: str, required_columns: List[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Safely loads a CSV file as a DataFrame and validates required columns.
    
    Args:
        file_path (str): Path to the CSV file
        required_columns (List[str]): List of column names that must be present
        
    Returns:
        Tuple[pd.DataFrame, Optional[str]]: A tuple with the DataFrame and error message if any
    """
    try:
        # Use explicit UTF-8 encoding to properly handle special characters like ñ
        df = pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        return pd.DataFrame(), f"❌ Error: Could not find the file '{file_path}'."
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), f"❌ Error: The file '{file_path}' is empty."
    except Exception as e:
        # Try with latin-1 encoding as fallback
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception:
            return pd.DataFrame(), f"❌ Error loading file '{file_path}': {str(e)}"
    
    # Check for required columns if specified
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return pd.DataFrame(), f"❌ Error: Missing required columns: {', '.join(missing_columns)}"
    
    return df, None


# --- Main Execution Functions ---

def ensure_output_dir(output_dir: str = "output") -> str:
    """
    Ensures the output directory exists, creates it if it doesn't.
    
    Args:
        output_dir (str): Name of the output directory
        
    Returns:
        str: Path to the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    return str(output_path)


def check_internal_sequence(args: Dict[str, Any]) -> None:
    """
    Checks for duplicates within the incoming list itself and assigns sequence numbers.
    """
    print(f"\n🔍 Checking for internal duplicates within the list ({args['incoming_file']})...")
    
    # Load the incoming DataFrame
    incoming_df, error = load_dataframe(
        args['incoming_file'], 
        [args['name_column']]
    )
    if error:
        print(error)
        return
    
    print(f"✅ Loaded {len(incoming_df)} entries from the list.")
    
    # Check for entity type column
    entity_type_column = args.get('entity_type_column')
    has_type_column = entity_type_column and entity_type_column in incoming_df.columns
    
    if has_type_column:
        print(f"📊 Using entity type column '{entity_type_column}' for dynamic entity typing")
        print(f"   (Empty values will be treated as vendors by default)")
    
    # Create standardized names for comparison
    if has_type_column:
        # Create clean names based on entity type in each row
        def standardize_based_on_type(row):
            # Handle "Group" column values (both "Vendor" and "Employee" in any case)
            # Empty values or missing values default to vendor
            row_type = str(row.get(entity_type_column, "")).strip().lower()
            
            if row_type == "" or row_type is None:
                return standardize_name(row[args['name_column']], "vendor")
            elif "employee" in row_type:
                return standardize_name(row[args['name_column']], "employee")
            else:  # Default to vendor for anything else
                return standardize_name(row[args['name_column']], "vendor")
            
        incoming_df['clean_name'] = incoming_df.apply(standardize_based_on_type, axis=1)
        
        # Store entity type info
        incoming_df['entity_type'] = incoming_df[entity_type_column].apply(
            lambda x: "employee" if "employee" in str(x).strip().lower() else "vendor"
        )
    else:
        # Use the default entity type for all entries
        incoming_df['clean_name'] = incoming_df[args['name_column']].apply(
            lambda x: standardize_name(x, args['entity_type'])
        )
        incoming_df['entity_type'] = args['entity_type']
    
    # Group by clean name and assign sequence numbers
    sequence_map = {}  # Will store clean_name -> sequence number mapping
    current_sequence = 1
    
    # Create result columns
    sequences = []
    match_scores = []
    matched_names = []
    
    # First pass to assign sequence numbers
    for idx, row in incoming_df.iterrows():
        clean_name = row['clean_name']
        
        if clean_name in sequence_map:
            # Already seen this name, use existing sequence
            sequence = sequence_map[clean_name]
            match_score = 100  # Perfect match with itself
            matched_name = sequence_map.get(f"{clean_name}_first_occurrence", "")
        else:
            # First time seeing this name, assign new sequence
            sequence = current_sequence
            sequence_map[clean_name] = sequence
            sequence_map[f"{clean_name}_first_occurrence"] = row[args['name_column']]
            current_sequence += 1
            match_score = 0  # No match (first occurrence)
            matched_name = ""
        
        sequences.append(sequence)
        match_scores.append(match_score)
        matched_names.append(matched_name)
    
    # Add the columns to the DataFrame
    incoming_df['Unique_Sequence'] = sequences
    incoming_df['Match_Score'] = match_scores
    incoming_df['Matched_Name_In_List'] = matched_names
    
    # Preserve the original order instead of sorting
    result_df = incoming_df
    
    # Drop temporary columns
    result_df = result_df.drop(columns=['clean_name', 'entity_type'], errors='ignore')
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.get('output_dir', 'output'))
    
    # Generate output filename
    base_filename = args.get('output_file') or "internal_sequence_report.csv"
    output_filename = os.path.join(output_dir, base_filename)
    
    # Save the results with explicit UTF-8 encoding
    result_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    # Count unique sequences and duplicates
    unique_sequences = len(set(sequences))
    duplicates = len(sequences) - unique_sequences
    
    print(f"✅ Processing Complete. Found {unique_sequences} unique entries and {duplicates} potential duplicates.")
    print(f"✅ Original order preserved in the output file.")
    print(f"💾 Results saved to '{output_filename}'")


def check_internal_duplicates(args: Dict[str, Any]) -> None:
    """
    Checks for duplicates within the main list.
    """
    print(f"\n🔍 Checking for duplicates within the main list ({args['main_file']})...")
    
    # Load the main DataFrame
    main_df, error = load_dataframe(
        args['main_file'], 
        [args['name_column'], args['code_column']]
    )
    if error:
        print(error)
        return
    
    print(f"✅ Loaded {len(main_df)} entries from main list.")
    
    # Find internal duplicates
    duplicates_df = find_duplicates_in_main_list(
        main_df,
        args['name_column'],
        args['code_column'],
        args['entity_type'],
        args['similarity']
    )
    
    if not duplicates_df.empty:
        # Ensure output directory exists
        output_dir = ensure_output_dir(args.get('output_dir', 'output'))
        
        # Generate output filename
        entity_label = "employee" if args['entity_type'] == "employee" else "vendor"
        base_filename = args.get('output_file') or f"{entity_label}_internal_duplicates.csv"
        output_filename = os.path.join(output_dir, base_filename)
        
        # Define column order for report
        report_columns = [
            args['name_column'], args['code_column'], 'Remark', 'Match_Score',
            'Duplicate_Of_Entity', 'Duplicate_Entity_Code'
        ]
        # Get remaining columns
        other_columns = [col for col in duplicates_df.columns 
                        if col not in report_columns]
        final_columns = report_columns + other_columns
        
        # Create report with ordered columns
        duplicates_df = duplicates_df[final_columns]
        
        # Save report with explicit UTF-8 encoding to handle special characters like ñ
        duplicates_df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"✅ Processing Complete. Found {len(duplicates_df)} potential duplicate entries.")
        print(f"💾 Results saved to '{output_filename}'")
    else:
        print("✅ No duplicates found. No output file created.")


def check_incoming_entries(args: Dict[str, Any]) -> None:
    """
    Checks incoming entries against the main list.
    """
    print(f"\n🔍 Checking incoming list ({args['incoming_file']}) against main list ({args['main_file']})...")
    
    # Load both DataFrames
    main_df, main_error = load_dataframe(
        args['main_file'], 
        [args['name_column'], args['code_column']]
    )
    if main_error:
        print(main_error)
        return
        
    incoming_df, incoming_error = load_dataframe(
        args['incoming_file'], 
        [args['name_column']]
    )
    if incoming_error:
        print(incoming_error)
        return
    
    print(f"✅ Loaded {len(main_df)} entries from main list and {len(incoming_df)} from incoming list.")
    
    # Check if entity type column is provided and exists
    entity_type_column = args.get('entity_type_column')
    if entity_type_column and entity_type_column in incoming_df.columns:
        print(f"📊 Using entity type column '{entity_type_column}' for dynamic entity typing")
        print(f"   (Empty values will be treated as vendors by default)")
    
    # Check incoming against main
    result_df = check_incoming_against_main(
        main_df,
        incoming_df,
        args['name_column'],
        args['code_column'],
        args['entity_type'],
        entity_type_column,
        args['similarity']
    )
    
    if not result_df.empty:
        # Ensure output directory exists
        output_dir = ensure_output_dir(args.get('output_dir', 'output'))
        
        # Generate output filename
        entity_label = "employee" if args['entity_type'] == "employee" else "vendor"
        base_filename = args.get('output_file') or f"{entity_label}_incoming_results.csv"
        output_filename = os.path.join(output_dir, base_filename)
        
        # Save the results with explicit UTF-8 encoding to handle special characters like ñ
        result_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        # Count duplicates and new entries
        duplicates = len(result_df[result_df['Match_Score'] > 0])
        new_entries = len(result_df) - duplicates
        
        print(f"✅ Processing Complete. Found {duplicates} potential duplicates and {new_entries} new entries.")
        print(f"💾 Results saved to '{output_filename}'")


# --- Command Line Interface ---

def parse_arguments():
    """
    Parse command line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Vendor and Employee Deduplication Tool for ERP Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check incoming vendors against main list:
    python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --entity-type vendor
        
  Check for duplicates within main list:
    python vendor_dedup_manager.py --check-internal --main-file main_list.csv --entity-type vendor
        
  Check incoming employees with custom column names:
    python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_employees.csv --entity-type employee --name-column "Employee Name" --code-column "Emp ID"
        
  Process mixed list with the "Group" column:
    python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file mixed_list.csv
        
  Check a single list for internal duplicates and assign sequence numbers:
    python vendor_dedup_manager.py --sequence-check --incoming-file new_list.csv  # No main-file needed
        
  Specify a custom output directory:
    python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --output-dir "my_reports"
        """
    )
    
    # Main action group (required: one of these)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--check-incoming", 
        action="store_true", 
        help="Check incoming entries against main list"
    )
    action_group.add_argument(
        "--check-internal", 
        action="store_true",
        help="Check for duplicates within main list"
    )
    action_group.add_argument(
        "--sequence-check",
        action="store_true",
        help="Check for duplicates within a single list and assign sequence numbers"
    )
    
    # File arguments
    parser.add_argument(
        "--main-file", 
        help="Path to the main list CSV file (required with --check-incoming and --check-internal)"
    )
    parser.add_argument(
        "--incoming-file", 
        help="Path to the incoming entries CSV file (required with --check-incoming or --sequence-check)"
    )
    parser.add_argument(
        "--output-file", 
        help="Filename for the output CSV file (optional, will use default naming otherwise)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where all output files will be saved (default: 'output')"
    )
    
    # Column names
    parser.add_argument(
        "--name-column", 
        default="Vendor Name",
        help="Column name containing the entity names (default: 'Vendor Name')"
    )
    parser.add_argument(
        "--code-column", 
        default="New Vendor Code",
        help="Column name containing the entity codes (default: 'New Vendor Code')"
    )
    
    # Entity type
    parser.add_argument(
        "--entity-type", 
        choices=["vendor", "employee"],
        default="vendor",
        help="Default type of entities to process (default: vendor)"
    )
    
    parser.add_argument(
        "--entity-type-column",
        default="Group",
        help="Column in incoming list specifying entity type ('Vendor' or 'Employee') for each row (default: 'Group')"
    )
    
    # Matching settings
    parser.add_argument(
        "--similarity", 
        type=int,
        default=90,
        help="Similarity threshold (0-100) to consider a match (default: 90)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on chosen action
    if args.check_incoming and not args.incoming_file:
        parser.error("--check-incoming requires --incoming-file")
    
    if args.sequence_check and not args.incoming_file:
        parser.error("--sequence-check requires --incoming-file")
        
    if (args.check_incoming or args.check_internal) and not args.main_file:
        parser.error("--check-incoming and --check-internal require --main-file")
    
    return args


def main():
    """
    Main entry point of the script.
    """
    args = vars(parse_arguments())
    
    print("\n📋 Vendor/Employee Deduplication Tool")
    print("=" * 50)
    
    if args['check_internal']:
        check_internal_duplicates(args)
    elif args['check_incoming']:
        check_incoming_entries(args)
    elif args['sequence_check']:
        check_internal_sequence(args)
    
    print("\n✨ Operation completed successfully.\n")


if __name__ == "__main__":
    main()