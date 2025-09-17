import shutil
import zipfile
from pathlib import Path

import yaml
from src.core.data_preprocessing import WordParser
from src.utils.corrupted_docx_analyzer import DocxXMLAttributeAnalyzer
from src.utils.docx_graphics_datafixer import DocxGraphicsDataFixer
from src.utils.file_manager import unzip_docx_to_folder


def main():
    # Load the YAML file
    with open("../config/benchmark_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Create parser instance
    word_parser = WordParser(
        input_folder_path=config["loading"]["input_folder"],
        output_folder_path=config["loading"]["output_folder"],
    )
    # Parse all files in a folder
    skipped_files = word_parser.parse_all_files()
    print(f"Skipped files: {skipped_files}")


if __name__ == "__main__":
    main()