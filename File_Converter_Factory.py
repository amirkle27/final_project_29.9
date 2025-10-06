from pathlib import Path
from File_Converter import FileConverter, ExcelConverter, CSVConverter, JSONConverter

class FileConverterFactory:
    """
    Factory class for creating appropriate FileConverter instances
    based on the file extension.

    This class follows the Factory Design Pattern, which allows the program
    to dynamically select the correct converter type (Excel, CSV, or JSON)
    at runtime based on the file's suffix.
    """
    @staticmethod
    def get(file_path:Path) -> FileConverter:
        """
        Returns an appropriate FileConverter instance for the given file.

        Args:
            file_path (Path): Path to the file that needs conversion.

        Returns:
            FileConverter: An instance of a subclass (ExcelConverter, CSVConverter, JSONConverter).

        Raises:
            ValueError: If the file type is not supported.
        """
        suffix = file_path.suffix.lower()

        if suffix in ['.xlsx', '.xls']:
            return ExcelConverter()
        elif suffix == '.csv':
            return CSVConverter()
        elif suffix == '.json':
            return JSONConverter()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

