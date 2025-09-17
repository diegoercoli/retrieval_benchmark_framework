import pandas as pd
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
from pathlib import Path


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


def process_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Process the dataset and return a dataframe with columns: complexity, question, ground_truth

    Args:
        dataset_path: Path to the Excel dataset file

    Returns:
        DataFrame with processed data containing ground truth document records
    """
    # Load the dataset
    df = pd.read_excel(dataset_path)

    # Filter for 'Analisi_Immagine' complexity
    complexity_column = "Complessità domanda"
    filtered_df = df[~df[complexity_column].str.contains('Analisi_Immagine', na=False, case=False)].copy()

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

    return final_df

# Example usage:
# df = process_dataset(Path("MrWolf_datasetdomande_IngMan.xlsm"))
# print(f"Processed {len(df)} rows")
# print(df.head())