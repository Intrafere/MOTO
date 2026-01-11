"""
Document ingestion pipeline.
Reads files, normalizes text, chunks at multiple configs, extracts metadata.
"""
import aiofiles
from pathlib import Path
from typing import List, Dict
import logging

from backend.shared.models import DocumentChunk
from backend.aggregator.ingestion.normalizer import normalize_text
from backend.aggregator.ingestion.chunker import chunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Document ingestion pipeline."""
    
    async def ingest_file(
        self,
        file_path: str,
        chunk_sizes: List[int] = None,
        is_user_file: bool = False
    ) -> Dict[int, List[DocumentChunk]]:
        """
        Ingest a file and return chunks at multiple sizes.
        
        Args:
            file_path: Path to file
            chunk_sizes: Sizes to chunk at (None = all configs)
            is_user_file: Whether this is a user-uploaded file
        
        Returns:
            Dict mapping chunk_size -> list of DocumentChunks
        """
        try:
            # Read file
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text = await f.read()
            
            # Normalize text
            normalized_text = normalize_text(text)
            
            # Get file name
            file_name = Path(file_path).name
            
            # Chunk at multiple sizes
            chunks_by_size = chunker.chunk_text(
                normalized_text,
                file_name,
                chunk_sizes,
                is_user_file
            )
            
            logger.info(f"Ingested {file_name}: {sum(len(chunks) for chunks in chunks_by_size.values())} total chunks")
            
            return chunks_by_size
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            raise
    
    async def ingest_text(
        self,
        text: str,
        source_name: str,
        chunk_sizes: List[int] = None,
        is_user_file: bool = False
    ) -> Dict[int, List[DocumentChunk]]:
        """
        Ingest raw text and return chunks at multiple sizes.
        
        Args:
            text: Text content
            source_name: Name to identify this content
            chunk_sizes: Sizes to chunk at (None = all configs)
            is_user_file: Whether this is user content
        
        Returns:
            Dict mapping chunk_size -> list of DocumentChunks
        """
        try:
            # Normalize text
            normalized_text = normalize_text(text)
            
            # Chunk at multiple sizes
            chunks_by_size = chunker.chunk_text(
                normalized_text,
                source_name,
                chunk_sizes,
                is_user_file
            )
            
            logger.info(f"Ingested {source_name}: {sum(len(chunks) for chunks in chunks_by_size.values())} total chunks")
            
            return chunks_by_size
            
        except Exception as e:
            logger.error(f"Failed to ingest text {source_name}: {e}")
            raise


# Global pipeline instance
ingestion_pipeline = IngestionPipeline()

