# Vendor Management & Duplicate Checking Tool - User Guide

This guide explains how to use the vendor duplicate checking system to manage your vendor data, check for duplicates, and maintain a clean vendor database.

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Common Tasks](#common-tasks)
   - [Check New Vendors Against Main List](#check-new-vendors-against-main-list)
   - [Check for Duplicates Within Main List](#check-for-duplicates-within-main-list) 
   - [Assign Unique Sequence Numbers](#assign-unique-sequence-numbers)
3. [Advanced Features](#advanced-features)
   - [Working with Mixed Entity Types](#working-with-mixed-entity-types)
   - [Customizing Output Files](#customizing-output-files)
4. [Troubleshooting](#troubleshooting)

## Installation & Setup

### Prerequisites
- Python 3.6 or higher
- Virtual environment (recommended)

### Setup
The tool is already set up in a virtual environment. To activate it:

```bash
# Navigate to your project directory
cd /path/to/Reviewer\ Python\ Codes/Duplicate\ Checking

# Activate the virtual environment
source venv/bin/activate
# You should see (venv) at the beginning of your command prompt
```

## Common Tasks

### Check New Vendors Against Main List

This checks a list of new vendors against your main vendor list to identify potential duplicates.

**Input needed:**
- `main_list.csv`: Your master vendor list
- `new_list.csv`: List of new vendors to check

**Command:**
```bash
python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv
```

**Output:**
- A CSV file in the `output` directory containing the incoming list with added columns:
  - `Remarks`: Indicates if each entry is a new vendor or potential duplicate
  - `Match_Score`: Similarity score (0-100)
  - `Matched_Vendor_In_Main`: The matched vendor name from the main list (if any)
  - `Existing_Vendor_Code`: The corresponding vendor code (if any)

### Check for Duplicates Within Main List

This checks your main list against itself to find potential duplicates within it.

**Input needed:**
- `main_list.csv`: Your master vendor list

**Command:**
```bash
python vendor_dedup_manager.py --check-internal --main-file main_list.csv
```

**Output:**
- A CSV file in the `output` directory containing potential duplicates with:
  - `Remark`: Indicates it's a potential duplicate
  - `Match_Score`: Similarity score (0-100)
  - `Duplicate_Of_Entity`: The matched vendor name
  - `Duplicate_Entity_Code`: The corresponding vendor code

### Assign Unique Sequence Numbers

This checks a single list for duplicates and assigns the same sequence number to similar entries.

**Input needed:**
- `new_list.csv`: List to check for internal duplicates

**Command:**
```bash
python vendor_dedup_manager.py --sequence-check --incoming-file new_list.csv
```

**Output:**
- A CSV file in the `output` directory containing the original list with added columns:
  - `Unique_Sequence`: A number grouping similar entries (identical entries get the same number)
  - `Match_Score`: 100 for matched entries, 0 for unique entries
  - `Matched_Name_In_List`: The first occurrence of the matched name (if any)

## Advanced Features

### Working with Mixed Entity Types

The tool can handle both vendors and employees in the same list using a `Group` column.

**How to use:**
1. Add a `Group` column to your CSV file
2. For each row, specify either "Vendor" or "Employee"
3. Run one of the commands as usual

**Example format:**
```csv
Vendor Name,Group
"ACME CORPORATION",Vendor
"DOE, JOHN M.",Employee
"WIDGET CO INC.",Vendor
"SMITH, JANE R.",Employee
```

**Command:**
```bash
python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file mixed_list.csv
```

The tool will:
- Apply vendor-specific standardization to vendors (removing Inc., Corp., etc.)
- Apply employee-specific standardization to employees (handling "LAST, FIRST M." format)
- Compare vendors only against vendors and employees only against employees

**Note:** If a row has an empty `Group` value, it will default to vendor.

### Customizing Output Files

You can customize output file names and locations:

**Custom output filename:**
```bash
python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --output-file my_results.csv
```

**Custom output directory:**
```bash
python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --output-dir reports
```

**Custom column names:**
```bash
python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --name-column "Company Name" --code-column "ID"
```

## Troubleshooting

### Special Characters Display Incorrectly

If you see issues with special characters like "ñ", ensure you're viewing the files with UTF-8 encoding. The tool is configured to handle special characters properly.

### Missing Duplicates

If the tool isn't finding expected duplicates:

1. **Adjust similarity threshold:**
   ```bash
   python vendor_dedup_manager.py --check-incoming --main-file main_list.csv --incoming-file new_list.csv --similarity 85
   ```
   Lower numbers (like 85) will find more potential matches but may include false positives.

2. **Check entity type handling:**
   Ensure your `Group` column is correctly populated with "Vendor" or "Employee" for each row.

### Viewing Help

For a complete list of options:
```bash
python vendor_dedup_manager.py --help
```

---

For any other questions or issues, please contact the system administrator.