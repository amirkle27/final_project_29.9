from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class FileConverter(ABC):
    """
    Abstract base class for file converters.

    Any subclass must implement the `convert_to_csv` method, which takes
    a file path and converts the file to a CSV format.
    """
    @abstractmethod
    def convert_to_csv(self, file_path:Path):
        """
        Converts the given file to a CSV file.

        Args:
            file_path (Path): The path of the input file.

        Returns:
            Path: The path to the resulting CSV file.
        """
        pass

class ExcelConverter(FileConverter):
    """
    Converter class for Excel files (.xlsx, .xls).

    Uses pandas to read Excel files and save them as CSV files.
    """
    def convert_to_csv(self, file_path: Path) -> Path:
        """
        Converts an Excel file to a CSV file.

        Args:
            file_path (Path): Path to the Excel file.

        Returns:
            Path: Path to the generated CSV file.

        Raises:
            ValueError: If the provided file is not a valid Excel file.
        """
        if not file_path.suffix.lower() in [".xlsx", ".xls"]:
            raise ValueError("Not a valid Excel file")
        df = pd.read_excel(file_path)
        csv_path = file_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path

class CSVConverter(FileConverter):
    """
    Converter class for CSV files.

    Since the input file is already a CSV, this class simply returns
    the same file path after validation.
    """
    def convert_to_csv(self, file_path: Path) -> Path:
        """
        Validates that the file is a CSV and returns its path.

        Args:
            file_path (Path): Path to the CSV file.

        Returns:
            Path: The same file path.

        Raises:
            ValueError: If the provided file is not a valid CSV file.
        """
        if file_path.suffix.lower() != ".csv":
            raise ValueError("Not a valid CSV file")
        return file_path

class JSONConverter(FileConverter):
    """
    Converter class for JSON files.

    Uses pandas to read JSON files and convert them into CSV format.
    """
    def convert_to_csv(self, file_path: Path) -> Path:
        """
        Converts a JSON file to a CSV file.

        Args:
            file_path (Path): Path to the JSON file.

        Returns:
            Path: Path to the generated CSV file.

        Raises:
            ValueError: If the provided file is not a valid JSON file.
        """
        if not file_path.suffix.lower() == ".json":
            raise ValueError("Not a valid JSON file")

        df = pd.read_json(file_path)
        csv_path = file_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path



