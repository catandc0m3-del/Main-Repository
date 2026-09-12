"""
data_quality_helpers.py

Helper functions for data quality operations in the DAP Individual Project.
This module provides functions for:
- Introducing data quality issues (for educational purposes)
- Generating data quality reports
- Detecting and handling outliers

Author: [Instructor Name]
Date: [Date]
Course: Data Acquisition and Processing (DAP)
"""

import pandas as pd
import numpy as np
from scipy import stats


def introduce_data_issues(df, random_seed=42):
    """
    Introduce various data quality issues into a clean dataset for educational purposes.
    
    This function deliberately introduces common data quality problems that students
    will learn to identify and fix during the data cleaning process.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The clean dataset to introduce issues into
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with introduced data quality issues
        
    Issues Introduced:
    ------------------
    1. Outliers - Extreme values (5-10 std dev from mean)
    2. Invalid values - Negative numbers, invalid month values
    3. Duplicates - Exact and near-duplicate records
    4. Missing values - Random NaN values
    5. Data type issues - String placeholders, numbers with commas, whitespace
    
    Example:
    --------
    >>> df_clean = pd.read_csv('clean_data.csv')
    >>> df_dirty = introduce_data_issues(df_clean, random_seed=42)
    >>> df_dirty.to_csv('dirty_data.csv', index=False)
    """
    
    np.random.seed(random_seed)
    df_dirty = df.copy()
    n_rows = len(df_dirty)
    
    print("Introducing data quality issues...")
    print("=" * 60)
    
    # Get numeric columns (excluding year, month which are identifiers)
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['year', 'month']]
    
    print(f"Numeric columns to process: {numeric_cols}")
    
    
    
    
    # =========================================================================
    # 1. OUTLIERS (Do this FIRST while data is still numeric)
    # =========================================================================
    print("\n1. Introducing OUTLIERS...")
    
    outlier_count = 0
    for col in numeric_cols[:3]:  # Apply to first 3 numeric columns
        if col in df_dirty.columns:
            # Calculate statistics on clean numeric data
            col_data = pd.to_numeric(df_dirty[col], errors='coerce')
            col_mean = col_data.mean()
            col_std = col_data.std()
            
            if pd.notna(col_mean) and pd.notna(col_std) and col_std > 0:
                # Add extreme high values (5-10 standard deviations above mean)
                n_outliers = 3
                valid_indices = df_dirty[col].dropna().index.tolist()
                if len(valid_indices) >= n_outliers:
                    outlier_indices = np.random.choice(valid_indices, n_outliers, replace=False)
                    for idx in outlier_indices:
                        multiplier = np.random.uniform(5, 10)
                        df_dirty.loc[idx, col] = col_mean + (multiplier * col_std)
                        outlier_count += 1
                
                # Add extreme low values (negative or very small)
                n_low_outliers = 2
                valid_indices = df_dirty[col].dropna().index.tolist()
                if len(valid_indices) >= n_low_outliers:
                    low_outlier_indices = np.random.choice(valid_indices, n_low_outliers, replace=False)
                    for idx in low_outlier_indices:
                        df_dirty.loc[idx, col] = col_mean * np.random.uniform(-0.5, 0.1)
                        outlier_count += 1
    
    print(f"   - Added approximately {outlier_count} outliers")
    
    # =========================================================================
    # 2. INVALID VALUES (Do while still numeric)
    # =========================================================================
    print("\n2. Introducing INVALID VALUES...")
    
    # 2a. Additional negative values where only positive expected
    for col in numeric_cols[:2]:
        if col in df_dirty.columns:
            valid_indices = df_dirty[df_dirty[col] > 0].index.tolist()
            n_negative = min(3, len(valid_indices))
            if n_negative > 0:
                neg_indices = np.random.choice(valid_indices, n_negative, replace=False)
                for idx in neg_indices:
                    df_dirty.loc[idx, col] = -abs(df_dirty.loc[idx, col])
    
    # 2b. Invalid month values
    if 'month' in df_dirty.columns:
        n_invalid = min(4, n_rows)
        invalid_indices = np.random.choice(df_dirty.index, n_invalid, replace=False)
        invalid_values = [0, 13, 14, -1]
        for i, idx in enumerate(invalid_indices):
            df_dirty.loc[idx, 'month'] = invalid_values[i % len(invalid_values)]
    
    print(f"   - Added negative values in positive-only columns")
    print(f"   - Added invalid month values")
    
    # =========================================================================
    # 3. DUPLICATE RECORDS
    # =========================================================================
    print("\n3. Introducing DUPLICATE RECORDS...")
    
    # 3a. Exact duplicates - copy some random rows
    n_exact_dupes = min(8, n_rows // 10)
    if n_exact_dupes > 0:
        dupe_indices = np.random.choice(df_dirty.index, n_exact_dupes, replace=False)
        exact_dupes = df_dirty.loc[dupe_indices].copy()
        df_dirty = pd.concat([df_dirty, exact_dupes], ignore_index=True)
    
    # 3b. Near-duplicates - same date but slightly different values
    n_near_dupes = min(5, n_rows // 15)
    if n_near_dupes > 0:
        near_dupe_indices = np.random.choice(df_dirty.index[:n_rows], n_near_dupes, replace=False)
        near_dupes = df_dirty.loc[near_dupe_indices].copy()
        
        # Modify numeric values slightly
        for col in numeric_cols:
            if col in near_dupes.columns:
                near_dupes[col] = near_dupes[col].apply(
                    lambda x: x * np.random.uniform(0.99, 1.01) if pd.notna(x) and isinstance(x, (int, float)) else x
                )
        df_dirty = pd.concat([df_dirty, near_dupes], ignore_index=True)
    
    print(f"   - Added {n_exact_dupes} exact duplicates")
    print(f"   - Added {n_near_dupes} near-duplicates")
    
    # Update row count after adding duplicates
    n_rows = len(df_dirty)
    
    # =========================================================================
    # 4. MISSING VALUES (Multiple patterns)
    # =========================================================================
    print("\n4. Introducing MISSING VALUES...")
    
    # 4a. Random NaN values in numeric columns (~5% of data)
    nan_count = 0
    for col in numeric_cols:
        if col in df_dirty.columns:
            mask = np.random.random(n_rows) < 0.05
            nan_count += mask.sum()
            df_dirty.loc[mask, col] = np.nan
    
    # 4b. Missing month_name values
    if 'month_name' in df_dirty.columns:
        mask = np.random.random(n_rows) < 0.02
        nan_count += mask.sum()
        df_dirty.loc[mask, 'month_name'] = np.nan
    
    print(f"   - Added approximately {nan_count} NaN values")
    
    # =========================================================================
    # 5. DATA TYPE ISSUES (Do AFTER numeric operations)
    # =========================================================================
    print("\n5. Introducing DATA TYPE ISSUES...")
    
    # 5a. String placeholders for missing values in total_arrivals
    if 'total_arrivals' in df_dirty.columns:
        # Convert to object type to allow mixed types
        df_dirty['total_arrivals'] = df_dirty['total_arrivals'].astype(object)
        
        # Add string placeholders
        valid_indices = df_dirty[df_dirty['total_arrivals'].notna()].index.tolist()
        n_placeholders = min(int(n_rows * 0.03), len(valid_indices))
        if n_placeholders > 0:
            placeholder_indices = np.random.choice(valid_indices, n_placeholders, replace=False)
            placeholders = ['NA', 'N/A', '-', '']
            for idx in placeholder_indices:
                df_dirty.loc[idx, 'total_arrivals'] = np.random.choice(placeholders)
        print(f"   - Added {n_placeholders} string placeholders for missing values")
        
        # 5b. Numbers with commas
        valid_numeric = df_dirty['total_arrivals'].apply(
            lambda x: isinstance(x, (int, float)) and pd.notna(x)
        )
        valid_indices = df_dirty[valid_numeric].index.tolist()
        n_comma = min(int(n_rows * 0.08), len(valid_indices))
        if n_comma > 0:
            comma_indices = np.random.choice(valid_indices, n_comma, replace=False)
            for idx in comma_indices:
                val = df_dirty.loc[idx, 'total_arrivals']
                try:
                    df_dirty.loc[idx, 'total_arrivals'] = f"{int(float(val)):,}"
                except:
                    pass
        print(f"   - Added {n_comma} numbers with comma formatting")
    
    # 5c. Whitespace and case issues in month_name
    if 'month_name' in df_dirty.columns:
        # Leading/trailing spaces
        valid_indices = df_dirty[df_dirty['month_name'].notna()].index.tolist()
        n_whitespace = min(int(n_rows * 0.08), len(valid_indices))
        if n_whitespace > 0:
            ws_indices = np.random.choice(valid_indices, n_whitespace, replace=False)
            for idx in ws_indices:
                val = df_dirty.loc[idx, 'month_name']
                if pd.notna(val):
                    df_dirty.loc[idx, 'month_name'] = f"  {val}  "
        
        # Case inconsistencies
        valid_indices = df_dirty[df_dirty['month_name'].notna()].index.tolist()
        n_case = min(int(n_rows * 0.05), len(valid_indices))
        if n_case > 0:
            case_indices = np.random.choice(valid_indices, n_case, replace=False)
            for idx in case_indices:
                val = df_dirty.loc[idx, 'month_name']
                if pd.notna(val) and isinstance(val, str):
                    val_clean = val.strip()
                    df_dirty.loc[idx, 'month_name'] = np.random.choice([val_clean.upper(), val_clean.lower()])
        
        print(f"   - Added whitespace and case inconsistencies in month_name")
    
    # =========================================================================
    # 6. SHUFFLE THE DATA
    # =========================================================================
    df_dirty = df_dirty.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print(f"Dirty dataset created: {df_dirty.shape[0]} rows × {df_dirty.shape[1]} columns")
    print(f"Original was: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df_dirty


def data_quality_report(df):
    """
    Generate a comprehensive data quality report for a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics
        
    Example:
    --------
    >>> df = pd.read_csv('data.csv')
    >>> report = data_quality_report(df)
    """
    print("DATA QUALITY REPORT")
    print("=" * 70)
    
    report = {}
    
    # Basic info
    print(f"\n1. BASIC INFORMATION")
    print("-" * 40)
    print(f"   Total Rows: {df.shape[0]}")
    print(f"   Total Columns: {df.shape[1]}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    report['total_rows'] = df.shape[0]
    report['total_columns'] = df.shape[1]
    
    # Data types
    print(f"\n2. DATA TYPES")
    print("-" * 40)
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}")
    
    # Missing values
    print(f"\n3. MISSING VALUES (NaN)")
    print("-" * 40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            print(f"   {col}: {missing[col]} ({missing_pct[col]}%)")
            has_missing = True
    if not has_missing:
        print("   No NaN values detected (but check for string placeholders!)")
    
    report['missing_values'] = missing.to_dict()
    
    # Check for string placeholders for missing values
    print(f"\n4. STRING PLACEHOLDERS FOR MISSING VALUES")
    print("-" * 40)
    placeholders = ['NA', 'N/A', 'na', 'n/a', '-', '', ' ', 'null', 'NULL', 'None', 'none', '.']
    found_placeholders = False
    placeholder_report = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            placeholder_counts = {}
            for p in placeholders:
                count = (df[col].astype(str).str.strip() == p).sum()
                if count > 0:
                    placeholder_counts[p if p != '' else '(empty)'] = count
            if placeholder_counts:
                print(f"   {col}: {placeholder_counts}")
                placeholder_report[col] = placeholder_counts
                found_placeholders = True
    if not found_placeholders:
        print("   No string placeholders found")
    
    report['string_placeholders'] = placeholder_report
    
    # Duplicates
    print(f"\n5. DUPLICATE RECORDS")
    print("-" * 40)
    exact_dupes = df.duplicated().sum()
    print(f"   Exact duplicates: {exact_dupes}")
    
    report['exact_duplicates'] = exact_dupes
    
    if 'date' in df.columns:
        date_dupes = df.duplicated(subset=['date'], keep=False).sum()
        print(f"   Records with duplicate dates: {date_dupes}")
        report['date_duplicates'] = date_dupes
    
    # Outliers (using IQR method) - only for truly numeric columns
    print(f"\n6. POTENTIAL OUTLIERS (IQR Method)")
    print("-" * 40)
    outlier_report = {}
    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0 and col not in ['year', 'month']:
                Q1 = numeric_col.quantile(0.25)
                Q3 = numeric_col.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((numeric_col < lower_bound) | (numeric_col > upper_bound)).sum()
                    if outliers > 0:
                        print(f"   {col}: {outliers} outliers (range: {lower_bound:.2f} to {upper_bound:.2f})")
                        outlier_report[col] = {
                            'count': outliers,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
        except:
            pass
    
    report['outliers'] = outlier_report
    
    # Invalid values
    print(f"\n7. INVALID VALUES")
    print("-" * 40)
    invalid_report = {}
    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0 and col not in ['year', 'month']:
                negatives = (numeric_col < 0).sum()
                if negatives > 0:
                    print(f"   {col}: {negatives} negative values")
                    invalid_report[col] = {'negative_values': negatives}
        except:
            pass
    
    if 'month' in df.columns:
        month_numeric = pd.to_numeric(df['month'], errors='coerce')
        invalid_months = ((month_numeric < 1) | (month_numeric > 12)).sum()
        if invalid_months > 0:
            print(f"   month: {invalid_months} invalid values (outside 1-12)")
            invalid_report['month'] = {'invalid_values': invalid_months}
    
    report['invalid_values'] = invalid_report
    
    # Data format issues
    print(f"\n8. DATA FORMAT ISSUES")
    print("-" * 40)
    format_report = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            col_issues = {}
            # Check for numbers with commas
            comma_nums = df[col].astype(str).str.contains(r'^[\d,]+$', regex=True, na=False).sum()
            if comma_nums > 0:
                print(f"   {col}: {comma_nums} values with comma formatting")
                col_issues['comma_formatting'] = comma_nums
            
            # Check for whitespace issues
            whitespace = df[col].astype(str).str.contains(r'^\s+|\s+$', regex=True, na=False).sum()
            if whitespace > 0:
                print(f"   {col}: {whitespace} values with leading/trailing whitespace")
                col_issues['whitespace_issues'] = whitespace
            
            if col_issues:
                format_report[col] = col_issues
    
    report['format_issues'] = format_report
    
    print("\n" + "=" * 70)
    
    return report


def detect_outliers_iqr(df, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    column : str
        Column name to check for outliers
        
    Returns:
    --------
    tuple
        (lower_bound, upper_bound, outlier_mask, outlier_count)
        
    Example:
    --------
    >>> lower, upper, mask, count = detect_outliers_iqr(df, 'total_arrivals')
    >>> print(f"Found {count} outliers")
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return lower_bound, upper_bound, outlier_mask, outlier_mask.sum()


def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    column : str
        Column name to check for outliers
    threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    tuple
        (outlier_mask, outlier_count)
        
    Example:
    --------
    >>> mask, count = detect_outliers_zscore(df, 'total_arrivals', threshold=3)
    >>> print(f"Found {count} outliers")
    """
    col_data = df[column].dropna()
    if len(col_data) == 0:
        return pd.Series(False, index=df.index), 0
    
    z_scores = np.abs(stats.zscore(col_data))
    outlier_mask = pd.Series(False, index=df.index)
    outlier_mask.loc[col_data.index] = z_scores > threshold
    
    return outlier_mask, outlier_mask.sum()


def cap_outliers(df, column, lower_bound=None, upper_bound=None, method='iqr'):
    """
    Cap outliers using specified bounds or IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
    column : str
        Column name to cap outliers
    lower_bound : float, optional
        Lower bound for capping. If None, calculated from method.
    upper_bound : float, optional
        Upper bound for capping. If None, calculated from method.
    method : str, default='iqr'
        Method to calculate bounds if not provided ('iqr' or 'percentile')
        
    Returns:
    --------
    pandas.Series
        Column with capped values
        
    Example:
    --------
    >>> df['total_arrivals'] = cap_outliers(df, 'total_arrivals')
    """
    if lower_bound is None or upper_bound is None:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR if lower_bound is None else lower_bound
            upper_bound = Q3 + 1.5 * IQR if upper_bound is None else upper_bound
        elif method == 'percentile':
            lower_bound = df[column].quantile(0.01) if lower_bound is None else lower_bound
            upper_bound = df[column].quantile(0.99) if upper_bound is None else upper_bound
    
    return df[column].clip(lower=lower_bound, upper=upper_bound)


# Make functions available when importing with *
__all__ = [
    'introduce_data_issues',
    'data_quality_report',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'cap_outliers'
]
