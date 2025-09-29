
from pathlib import Path


from File_Converter import FileConverter, ExcelConverter, CSVConverter, JSONConverter



class FileConverterFactory:
    @staticmethod
    def get(file_path:Path) -> FileConverter:
        suffix = file_path.suffix.lower()

        if suffix in ['.xlsx', '.xls']:
            return ExcelConverter()
        elif suffix == '.csv':
            return CSVConverter()
        elif suffix == '.json':
            return JSONConverter()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")




