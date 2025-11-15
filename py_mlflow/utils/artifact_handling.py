"""
Artifact compression and handling for large models.
"""

import os
import gzip
import shutil
from pathlib import Path
from typing import Any, Union, Optional


def should_compress(file_path: Union[str, Path], threshold_mb: float = 100.0) -> bool:
    """
    Determine if file should be compressed based on size.

    Args:
        file_path: Path to file
        threshold_mb: Size threshold in MB (default: 100MB)

    Returns:
        True if file exceeds threshold
    """
    if not os.path.exists(file_path):
        return False

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb > threshold_mb


def compress_artifact(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    remove_source: bool = False
) -> str:
    """
    Compress artifact using gzip.

    Args:
        source_path: Path to source file
        dest_path: Path to destination (default: source_path + '.gz')
        remove_source: Whether to remove source after compression

    Returns:
        Path to compressed file
    """
    source_path = Path(source_path)

    if dest_path is None:
        dest_path = Path(str(source_path) + '.gz')
    else:
        dest_path = Path(dest_path)

    with open(source_path, 'rb') as f_in:
        with gzip.open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove_source:
        os.remove(source_path)

    return str(dest_path)


def decompress_artifact(
    source_path: Union[str, Path],
    dest_path: Optional[Union[str, Path]] = None,
    remove_source: bool = False
) -> str:
    """
    Decompress gzipped artifact.

    Args:
        source_path: Path to compressed file
        dest_path: Path to destination (default: source_path without '.gz')
        remove_source: Whether to remove source after decompression

    Returns:
        Path to decompressed file
    """
    source_path = Path(source_path)

    if dest_path is None:
        # Remove .gz extension
        dest_path = Path(str(source_path).rstrip('.gz'))
    else:
        dest_path = Path(dest_path)

    with gzip.open(source_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove_source:
        os.remove(source_path)

    return str(dest_path)


def get_artifact_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get artifact size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        Size in MB
    """
    if not os.path.exists(file_path):
        return 0.0

    return os.path.getsize(file_path) / (1024 * 1024)
