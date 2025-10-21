import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging


def read_excel(path: Path, sheet_name: Union[str, int, None] = 0,
               header: Optional[int] = 0, **kwargs) -> pd.DataFrame:
    """
    Read an Excel file and return a pandas DataFrame.

    Args:
        path (Path): Path to the Excel file (.xlsx or .xls)
        sheet_name (str, int, None): Name or index of sheet to read.
                                   0 = first sheet (default)
                                   None = all sheets (returns dict)
        header (int, optional): Row to use as column names (0-indexed).
                              None = no header row
        **kwargs: Additional arguments passed to pd.read_excel()

    Returns:
        pd.DataFrame: DataFrame containing the Excel data

    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If the file is not a valid Excel format
        PermissionError: If the file is locked or access is denied
        Exception: For other pandas/Excel reading errors

    Example:
        >>> df = read_excel(Path("data/queries.xlsx"))
        >>> df = read_excel(Path("data/results.xlsx"), sheet_name="Results", header=1)
    """

    # Validate input path
    if not isinstance(path, Path):
        path = Path(path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # Check if it's a valid Excel file extension
    valid_extensions = {'.xlsx', '.xls', '.xlsm'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid Excel file format. Expected {valid_extensions}, got: {path.suffix}")

    # Check file size (warn for very large files)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 100:  # Files larger than 100MB
        logging.warning(f"Large Excel file detected: {file_size_mb:.1f}MB. Reading may take time.")

    try:
        # Default pandas Excel reading parameters for robustness
        default_kwargs = {
            'engine': 'openpyxl' if path.suffix.lower() in ['.xlsx', '.xlsm'] else 'xlrd',
            'na_values': ['', 'N/A', 'NA', 'n/a', 'null', 'NULL', '#N/A'],
            'keep_default_na': True,
        }

        # Merge user kwargs with defaults (user kwargs take precedence)
        merged_kwargs = {**default_kwargs, **kwargs}

        # Read the Excel file
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header,
            **merged_kwargs
        )

        # Log basic info about the loaded data
        if isinstance(df, pd.DataFrame):
            logging.info(f"Successfully loaded Excel file: {path}")
            logging.info(f"Shape: {df.shape} (rows, columns)")
            if not df.empty:
                logging.info(f"Columns: {list(df.columns)}")
        else:
            # Multiple sheets returned as dict
            logging.info(f"Successfully loaded Excel file with {len(df)} sheets: {path}")

        return df

    except PermissionError as e:
        raise PermissionError(
            f"Permission denied reading Excel file: {path}. "
            f"File may be open in another application. Error: {e}"
        )

    except pd.errors.EmptyDataError:
        logging.warning(f"Excel file is empty: {path}")
        return pd.DataFrame()  # Return empty DataFrame

    except Exception as e:
        # Handle various pandas/Excel specific errors
        error_type = type(e).__name__
        error_msg = str(e)

        # Provide more specific error messages for common issues
        if "No such file or directory" in error_msg:
            raise FileNotFoundError(f"Excel file not found: {path}")
        elif "Unsupported format" in error_msg or "Excel file format cannot be determined" in error_msg:
            raise ValueError(f"Unsupported or corrupted Excel file: {path}")
        elif "worksheet" in error_msg.lower() and sheet_name:
            raise ValueError(f"Sheet '{sheet_name}' not found in Excel file: {path}")
        else:
            raise Exception(f"Error reading Excel file {path}. {error_type}: {error_msg}")


def read_excel_with_validation(path: Path, required_columns: Optional[list] = None,
                               sheet_name: Union[str, int, None] = 0) -> pd.DataFrame:
    """
    Read Excel file with column validation for your benchmark dataset.

    Args:
        path (Path): Path to Excel file
        required_columns (list): List of required column names
        sheet_name: Sheet to read

    Returns:
        pd.DataFrame: Validated DataFrame

    Raises:
        ValueError: If required columns are missing
    """

    df = read_excel(path, sheet_name=sheet_name)

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {path}: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

    # Basic data quality checks
    if df.empty:
        logging.warning(f"Loaded DataFrame is empty: {path}")

    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        logging.info(f"Found {empty_rows} completely empty rows, removing them.")
        df = df.dropna(how='all').reset_index(drop=True)

    return df


# Example usage for your benchmark dataset
def load_benchmark_dataset(dataset_path: Path, required_columns: list ) -> pd.DataFrame:
    """
    Load your specific benchmark dataset with expected columns.

    Args:
        dataset_path (Path): Path to the Excel file
        required_columns (list): List of required column names.
                               Default: ['id', 'query', 'target_subsections']

    Default expected columns for benchmark:
    - id: Query identifier
    - query: The question text
    - target_subsections: Ground truth subsections (could be pipe-separated)
    - document: Source document (optional)
    """

    if required_columns is None:
        required_columns = ['id', 'query', 'target_subsections']

    try:
        df = read_excel_with_validation(
            dataset_path,
            required_columns=required_columns
        )

        # Parse target_subsections if they're pipe-separated strings
        if 'target_subsections' in df.columns:
            # Convert string "subsec1|subsec2|subsec3" to list
            df['target_subsections'] = df['target_subsections'].apply(
                lambda x: str(x).split('|') if pd.notna(x) else []
            )

        logging.info(f"Loaded benchmark dataset with {len(df)} queries")
        return df

    except Exception as e:
        logging.error(f"Failed to load benchmark dataset: {e}")
        raise