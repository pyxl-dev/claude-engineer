from tools.base import BaseTool
import os
from pathlib import Path
from typing import List, Dict

class DirectoryListTool(BaseTool):
    name = "directorylisttool"
    description = '''
    Lists all files and directories within a specified parent directory.
    Can recursively list subdirectories and provides detailed metadata.
    Excludes system and hidden files by default.
    Returns a structured list of contents with file/directory information.
    '''
    input_schema = {
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Path to the directory to list contents from"
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to recursively list subdirectories",
                "default": False
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files and directories",
                "default": False
            }
        },
        "required": ["directory_path"]
    }

    def execute(self, **kwargs) -> str:
        directory_path = kwargs.get("directory_path")
        recursive = kwargs.get("recursive", False)
        include_hidden = kwargs.get("include_hidden", False)

        try:
            path = Path(directory_path)
            if not path.exists():
                return f"Error: Directory '{directory_path}' does not exist"
            if not path.is_dir():
                return f"Error: '{directory_path}' is not a directory"

            result = self._list_directory(path, recursive, include_hidden)
            return self._format_output(result)

        except Exception as e:
            return f"Error listing directory contents: {str(e)}"

    def _list_directory(self, path: Path, recursive: bool, include_hidden: bool) -> List[Dict]:
        contents = []
        try:
            for entry in sorted(path.iterdir()):
                # Skip hidden files unless explicitly included
                if not include_hidden and entry.name.startswith('.'):
                    continue

                item = {
                    'name': entry.name,
                    'path': str(entry),
                    'type': 'directory' if entry.is_dir() else 'file',
                    'size': entry.stat().st_size if entry.is_file() else None,
                    'modified': entry.stat().st_mtime
                }

                if recursive and entry.is_dir():
                    item['contents'] = self._list_directory(entry, recursive, include_hidden)

                contents.append(item)

        except PermissionError:
            contents.append({
                'name': path.name,
                'path': str(path),
                'type': 'error',
                'error': 'Permission denied'
            })

        return contents

    def _format_output(self, contents: List[Dict]) -> str:
        def format_entry(entry: Dict, level: int = 0) -> str:
            indent = "  " * level
            result = f"{indent}{entry['name']} ({entry['type']})"
            
            if entry['type'] == 'file':
                size = entry.get('size', 0)
                if size is not None:
                    result += f" - {self._format_size(size)}"
            
            if entry.get('error'):
                result += f" - Error: {entry['error']}"
            
            if entry.get('contents'):
                result += "\n" + "\n".join(format_entry(item, level + 1) 
                                         for item in entry['contents'])
            
            return result

        return "\n".join(format_entry(entry) for entry in contents)

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"