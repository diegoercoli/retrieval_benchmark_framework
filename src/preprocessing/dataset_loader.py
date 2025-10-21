import pandas as pd
import re
from pathlib import Path
from typing import List

from src.core.configuration import EvaluationConfiguration, PreprocessConfiguration
from src.models import (
    DatasetInput,
    Query,
    GroundTruth,
    HierarchicalMetadata,
    ConfidenceLevel,
    ComplexityQuery
)


def parse_ground_truths(cell_value, confidence_value) -> List[GroundTruth]:
    """
    Parse ground truth records from a cell into list of GroundTruth objects.

    Args:
        cell_value: Cell content with document and section information
        confidence_value: Confidence level string ('Low', 'Medium', 'High')

    Returns:
        List of GroundTruth objects
    """
    if pd.isna(cell_value):
        return []

    # Map confidence string to enum
    confidence_map = {
        'Low': ConfidenceLevel.LOW,
        'Medium': ConfidenceLevel.MEDIUM,
        'High': ConfidenceLevel.HIGH
    }
    #confidence = confidence_map.get(confidence_value, ConfidenceLevel.MEDIUM)
    #old_confidence = confidence_value
    ground_truths = []
    lines = str(cell_value).strip().split('\n')
    confidence_values = []
    if ',' in confidence_value:
        confidence_values = confidence_value.strip().split(',')
        confidence_values = [confidence_map[val.strip()] for val in confidence_values]
    else:
        confidence_values = [confidence_map[confidence_value.strip()]]
    #print(f"{old_confidence} ===> {confidence_values}")
    for index, line in enumerate(lines):
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

            hierarchical_metadata = None
            if section_match:
                section_number = section_match.group(1)
                section_title = section_match.group(2).strip()

                # Calculate depth from section number (e.g., "3.2.1" has depth 3)
                depth = len(section_number.split('.'))

                hierarchical_metadata = HierarchicalMetadata(
                    id_section=section_number,
                    section_title=section_title,
                    depth=depth
                )

            ground_truths.append(GroundTruth(
                filename=filename,
                hierarchical_metadata=hierarchical_metadata,
                confidence=confidence_values[index]
            ))

    return ground_truths


def load_dataset(
        dataset_path: Path,
        dataset_name: str
) -> DatasetInput:
    """
       Process the dataset and return a DatasetInput object for REST API submission.

       Args:
           dataset_path: Path for the dataset Excel file
           dataset_name: Name for the dataset (default: "benchmark_dataset")

       Returns:
           DatasetInput object ready for API submission via create_dataset()
    """
    # Extract parameters from config

    # Complexity mapping from Italian to API enum values
    complexity_mapping = {
        'Descrizione_Testuale': ComplexityQuery.TEXTUAL_DESCRIPTION,
        'Analisi_Immagine': ComplexityQuery.IMAGE_ANALYSIS,
        'Analisi_Tabella': ComplexityQuery.TABLE_ANALYSIS,
        'text': ComplexityQuery.TEXTUAL_DESCRIPTION,
        'image': ComplexityQuery.IMAGE_ANALYSIS,
        'table': ComplexityQuery.TABLE_ANALYSIS
    }

    # Load the dataset
    df = pd.read_excel(dataset_path)

    # Get complexity column name
    complexity_column = "Complessità domanda"

    # Apply filters
    filtered_df = df.copy()


    # 2. Filter out noisy queries
    if 'Noise' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['Noise'].isna() | (filtered_df['Noise'] == '')
            ].copy()
        print(f"Filtered by noise: {len(filtered_df)} queries remaining")

    # Add header to questions
    ''' 
    def add_header(row):
        modello_apparato = row["Modello Apparato"] if pd.notna(row["Modello Apparato"]) else "N/A"
        customer = row["Customer"] if pd.notna(row["Customer"]) else "N/A"
        original = row["Domanda rielaborata (ST)"] if pd.notna(row["Domanda rielaborata (ST)"]) else ""
        header = f"Apparato: {modello_apparato}, Customer: {customer}. "
        return header + str(original)

    filtered_df["Domanda rielaborata (ST)"] = filtered_df.apply(add_header, axis=1)
    '''


    # Build list of Query objects
    queries = []

    for idx, row in filtered_df.iterrows():
        # Map complexity to API enum
        complexity_value = row["Complessità domanda"]
        complexity_enum = complexity_mapping.get(
            complexity_value,
            ComplexityQuery.TEXTUAL_DESCRIPTION
        )

        # Parse ground truths
        ground_truths = parse_ground_truths(
            row["Nome documento risposta"],
            row["Confidenza documento"]
        )

        # Create Query object
        query = Query(
            position_id=idx + 1,  # 1-indexed position
            prompt=row["Domanda rielaborata (ST)"],
            device=row.get("Modello Apparato") if pd.notna(row.get("Modello Apparato")) else None,
            customer=row.get("Customer") if pd.notna(row.get("Customer")) else None,
            complexity=complexity_enum,
            ground_truths=ground_truths
        )

        queries.append(query)

    print(f"Final dataset: {len(queries)} queries prepared for API submission")

    # Create and return DatasetInput object
    return DatasetInput(
        dataset_name=dataset_name,
        queries=queries
    )