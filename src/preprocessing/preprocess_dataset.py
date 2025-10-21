import pandas as pd
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
from pathlib import Path

from src.core.configuration import EvaluationConfiguration, PreprocessConfiguration


class ConfidenceLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class GroundTruthDocumentRecord:
    filename: str
    section_number: str
    section_title: str
    confidence: ConfidenceLevel


def parse_document_records(cell_value, confidence_value) -> List[GroundTruthDocumentRecord]:
    """Parse document records from a cell into list of GroundTruthDocumentRecord objects"""
    if pd.isna(cell_value):
        return []

    # Map confidence string to enum
    confidence_map = {
        'Low': ConfidenceLevel.LOW,
        'Medium': ConfidenceLevel.MEDIUM,
        'High': ConfidenceLevel.HIGH
    }
    confidence = confidence_map.get(confidence_value, ConfidenceLevel.MEDIUM)

    records = []
    lines = str(cell_value).strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract filename from path
        if '\\' in line:
            path_part = line.split(',')[0] if ',' in line else line
            filename = path_part.split('\\')[-1]

            # Extract section info after comma or filename
            if ',' in line:
                section_part = line.split(',', 1)[1].strip()
            else:
                match = re.search(r'\.(docx|pdf|xlsx)(.+)', line)
                section_part = match.group(2) if match else ""

            # Parse section number and title
            section_match = re.match(r'^(\d+(?:\.\d+)*)\s*(.*)$', section_part.strip())
            if section_match:
                section_number = section_match.group(1)
                section_title = section_match.group(2).strip()
            else:
                section_number = ""
                section_title = section_part.strip()

            records.append(GroundTruthDocumentRecord(
                filename=filename,
                section_number=section_number,
                section_title=section_title,
                confidence=confidence
            ))

    return records


def process_dataset(evaluation_config: EvaluationConfiguration, preprocess_config: PreprocessConfiguration) -> pd.DataFrame:
    """
    Process the dataset and return a dataframe with columns: complexity, question, ground_truth.

    Args:
        config: EvaluationConfiguration object containing dataset_path and allowed_query_complexities.
        preprocess_config: PreprocessConfiguration object specifying text preprocessing options.

    Returns:
        DataFrame with processed data containing ground truth document records.
        The 'question' column is lowercased if preprocess_config.lowercase is True.
    """
    # Extract parameters from config
    dataset_path = evaluation_config.dataset_path
    allowed_query_complexities = evaluation_config.allowed_query_complexities

    # Complexity mapping
    complexity_mapping = {
        'text': 'Descrizione_Testuale',
        'image': 'Analisi_Immagine',
        'table': 'Analisi_Tabella'
    }

    # Load the dataset
    df = pd.read_excel(dataset_path)

    # Get complexity column name
    complexity_column = "Complessità domanda"

    # Apply filters
    filtered_df = df.copy()

    # 1. Filter by allowed query complexities
    if allowed_query_complexities:
        # Convert allowed complexities to dataset values
        allowed_values = [complexity_mapping.get(c, c) for c in allowed_query_complexities]

        # Filter rows where complexity matches allowed values
        filtered_df = filtered_df[
            filtered_df[complexity_column].isin(allowed_values)
        ].copy()

        print(f"Filtered by complexity: {len(df)} -> {len(filtered_df)} queries")

    # 2. Filter out noisy queries
    if 'Noise' in filtered_df.columns:
        # Keep only rows where noise is empty/null
        filtered_df = filtered_df[
            filtered_df['Noise'].isna() | (filtered_df['Noise'] == '')
            ].copy()

        print(f"Filtered by noise: {len(filtered_df)} queries remaining")

    # Add header to questions
    def add_header(row):
        modello_apparato = row["Modello Apparato"] if pd.notna(row["Modello Apparato"]) else "N/A"
        customer = row["Customer"] if pd.notna(row["Customer"]) else "N/A"
        original = row["Domanda rielaborata (ST)"] if pd.notna(row["Domanda rielaborata (ST)"]) else ""
        header = f"Apparato: {modello_apparato}, Customer: {customer}. "
        return header + str(original)

    filtered_df["Domanda rielaborata (ST)"] = filtered_df.apply(add_header, axis=1)

    # Select and rename relevant columns
    column_mapping = {
        "Complessità domanda": "complexity",
        "Domanda rielaborata (ST)": "question"
    }

    final_df = filtered_df[list(column_mapping.keys()) + ["Nome documento risposta", "Confidenza documento"]].rename(
        columns=column_mapping)

    # Parse document records into ground truth objects
    final_df['ground_truth'] = final_df.apply(
        lambda row: parse_document_records(row["Nome documento risposta"], row["Confidenza documento"]),
        axis=1
    )

    # Drop the intermediate columns and keep only the final three
    final_df = final_df[['complexity', 'question', 'ground_truth']]

    # Convert question text to lowercase
    if preprocess_config.lowercase:
        final_df['question'] = final_df['question'].str.lower()

    print(f"Final dataset: {len(final_df)} queries")

    return final_df

# Example usage:
# df = process_dataset(Path("MrWolf_datasetdomande_IngMan.xlsm"))
# print(f"Processed {len(df)} rows")
# print(df.head())