from tools.base import BaseTool
import json
from typing import Union, List
from pathlib import Path

class JsonReaderTool(BaseTool):
    name = "jsonreadertool"
    description = '''
    Reads and parses JSON files from provided file paths.
    Can handle both single files and multiple files.
    Returns parsed JSON content.
    Handles common JSON parsing errors gracefully.
    '''
    input_schema = {
        "type": "object",
        "properties": {
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of paths to JSON files to read"
            }
        },
        "required": ["file_paths"]
    }

    def execute(self, **kwargs) -> str:
        file_paths = kwargs.get('file_paths', [])
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        results = {}
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    results[file_path] = {"error": f"File not found: {file_path}"}
                    continue
                
                with open(path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    results[file_path] = content
                    
            except json.JSONDecodeError as e:
                results[file_path] = {"error": f"JSON parsing error in {file_path}: {str(e)}"}
            except Exception as e:
                results[file_path] = {"error": f"Error reading {file_path}: {str(e)}"}

        if len(results) == 1:
            return json.dumps(list(results.values())[0], indent=2)
        return json.dumps(results, indent=2)