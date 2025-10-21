from pathlib import Path

import yaml

from src.core.experiment_factory import BenchmarkFactory
from src.preprocessing.data_preprocessing import WordParser


def main():
    """
    Main entry point for the RAG benchmarking system.

    This function:
    1. Optionally runs document preprocessing (DOCX to JSON conversion)
    2. Runs the complete benchmarking pipeline using BenchmarkFactory
    """
    try:
        print("Starting RAG Benchmark System")

        # Configuration file path
        config_path = Path("../config/benchmark_config.yaml").resolve()

        # Optional: Run document preprocessing if needed
        # This converts DOCX files to JSON format for faster processing
        #run_preprocessing = input("Run document preprocessing? (y/n): ").lower().strip() == 'y'

        print("Running document preprocessing...")
        # Load configuration to get paths
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create parser instance and process documents
        word_parser = WordParser(
            input_folder_path=config["loading"]["input_folder"],
            output_folder_path=config["loading"]["output_folder"],
        )

        # Parse all files in the input folder
        skipped_files = word_parser.parse_all_files()
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files: {skipped_files}")
        else:
            print("All documents processed successfully")
        # Run the complete benchmarking pipeline
        print("Starting benchmark factory...")
        benchmark_factory = BenchmarkFactory(config_path)
        benchmark_factory.start()

        print("Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()