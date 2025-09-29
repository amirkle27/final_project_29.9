from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class FileConverter(ABC):
    @abstractmethod
    def convert_to_csv(self, file_path:Path):
        pass

class ExcelConverter(FileConverter):
    def convert_to_csv(self, file_path: Path) -> Path:
        if not file_path.suffix.lower() in [".xlsx", ".xls"]:
            raise ValueError("Not a valid Excel file")
        df = pd.read_excel(file_path)
        csv_path = file_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path

class CSVConverter(FileConverter):
    def convert_to_csv(self, file_path: Path) -> Path:
        if file_path.suffix.lower() != ".csv":
            raise ValueError("Not a valid CSV file")
        return file_path

class JSONConverter(FileConverter):
    def convert_to_csv(self, file_path: Path) -> Path:
        if not file_path.suffix.lower() == ".json":
            raise ValueError("Not a valid JSON file")

        df = pd.read_json(file_path)
        csv_path = file_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
