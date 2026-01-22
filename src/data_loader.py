
from pathlib import Path
from typing import List, Any, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
import os

def load_documents(data_source: Union[str, List[str]], source_type: str = "files") -> List[Any]:


    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.json': JSONLoader
    }

    documents = []
    file_paths = []

    # Handle directory or files
    if source_type == "directory":
        data_path = Path(data_source)
        for file_path in data_path.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_paths.append(file_path)
    else:  # source_type == "files"
        file_paths = [Path(fp) for fp in data_source]

    # Load each file
    for file_path in file_paths:
        ext = file_path.suffix.lower()
        if ext in loaders:
            loader_class = loaders[ext]
            try:
                loader = loader_class(str(file_path))
                loaded_docs = loader.load()

                # Add file metadata
                for doc in loaded_docs:
                    doc.metadata.update({
                        'source_type': 'file',
                        'file_path': str(file_path),
                        'file_name': file_path.name
                    })

                documents.extend(loaded_docs)
                print(f" Loaded: {file_path.name}")
            except Exception as e:
                print(f" Error loading {file_path.name}: {e}")
        else:
            print(f" Unsupported file type: {file_path.suffix}")

    return documents

def create_video_metadata_document(video_info: dict) -> List[Any]:


    from langchain_core.documents import Document

    content = f"""Video File: {video_info.get('filename', 'Unknown')}
File Size: {video_info.get('filesize', 0)} bytes
Duration: {video_info.get('duration', 0):.2f} seconds
Transcript Status: {'Available' if video_info.get('has_transcript') else 'Not available'}"""

    document = Document(
        page_content=content,
        metadata={
            'source_type': 'video',
            'file_name': video_info.get('filename', 'Unknown'),
            'file_path': video_info.get('filepath', ''),
            'duration': video_info.get('duration', 0),
            'has_transcript': video_info.get('has_transcript', False),
            'transcript_source': video_info.get('transcript_source', 'unknown')
        }
    )

    return [document]

