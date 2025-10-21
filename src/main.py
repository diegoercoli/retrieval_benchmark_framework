from pathlib import Path

import yaml

from src.core.benchmark_setup import BenchmarkSetup
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
        config_path = "../config/benchmark_config.yaml"
        benchmark_setup = BenchmarkSetup(config_path)
        benchmark_setup.initialize()

        print("Benchmark completed successfully!")

    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        raise


if __name__ == "__main__":
    main()