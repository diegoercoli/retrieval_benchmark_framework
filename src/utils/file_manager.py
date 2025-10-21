import re
import zipfile
from pathlib import Path
from typing import List, Set
from pathlib import Path
from typing import List
import pandas as pd
from docling_core.types import DoclingDocument


def unzip_docx_to_folder(docx_path: Path, extract_to: Path) -> Path:
    """
    Unzips a DOCX file to a folder.
    Returns the path to the folder.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "r") as z:
        z.extractall(extract_to)
    return extract_to

def scan_folder(inputFolderPath: Path, extension: str = "") -> Set[str]:
    """
    Recursively scan a folder and return all file paths matching a given extension.

    Args:
        inputFolderPath (str): Path to the folder to scan.
        extension (str, optional): File extension filter (e.g., ".docx").
            - If empty or invalid (not starting with "."), all files are scanned.

    Returns:
        Set[str]: A set of file paths (as strings) that match the given extension.
                  Temporary files (e.g., Word lock files starting with "~$") are excluded.

    Raises:
        FileNotFoundError: If the input folder does not exist.

    Example:
        >>> scan_folder("docs", ".txt")
        {'docs/notes.txt', 'docs/todo.txt'}
    """
    # Convert the input string to a Path object for easier path operations
    folder_path = inputFolderPath

    # Check if the folder exists, otherwise raise an error
    if not folder_path.exists():
        raise FileNotFoundError(f"Error: Folder '{inputFolderPath}' does not exist.")

    # Decide on the search pattern:
    # - If no extension is provided or it's invalid (not starting with "."),
    #   then match all files ("*").
    # - Otherwise, match files with the given extension (e.g., "*.docx").
    if not extension or not extension.startswith("."):
        pattern = "*"
    else:
        pattern = f"*{extension}"

    # Recursively find all files in the folder (and subfolders) that match the pattern
    files = list(folder_path.rglob(pattern))

    # Return the file paths as strings, but exclude temporary files
    # (e.g., Word lock files starting with "~$")
    return {str(f) for f in files if not f.name.startswith("~$")}


def sanitize_filename(name: str, replacement: str = "_") -> str:
    """
    Remove or replace characters that are not allowed in filenames.

    Args:
        name (str): The original string.
        replacement (str): Replacement character for invalid ones (default "_").

    Returns:
        str: A safe filename string.
    """
    # Characters not allowed on Windows
    invalid_chars = r'[<>:"/\\|?*]'
    # Replace them with replacement
    sanitized = re.sub(invalid_chars, replacement, name)

    # Remove control chars (ASCII < 32)
    sanitized = re.sub(r"[\x00-\x1f]", replacement, sanitized)

    # Avoid reserved names in Windows (CON, PRN, etc.)
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if sanitized.upper() in reserved:
        sanitized = f"_{sanitized}_"

    return sanitized.strip()


def sanitize_collection_name( name: str) -> str:
    """
    Sanitize collection name to meet Weaviate's requirements.

    Weaviate class names must:
    - Start with a capital letter
    - Only contain alphanumeric characters
    - No special characters, hyphens, or underscores
    """
    # Remove special characters and replace with empty string
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', name)

    # Ensure it starts with a capital letter
    if sanitized and not sanitized[0].isupper():
        sanitized = sanitized[0].upper() + sanitized[1:]

    # If empty after sanitization, provide a default
    if not sanitized:
        sanitized = "DefaultCollection"

    # Ensure it's not too long (Weaviate has limits)
    if len(sanitized) > 50:
        sanitized = sanitized[:50]

    return sanitized
