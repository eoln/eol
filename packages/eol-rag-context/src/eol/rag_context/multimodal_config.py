"""
Configuration for multimodal knowledge graph features.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ProcessingMode(Enum):
    """Processing modes for multimodal content."""

    MINIMAL = "minimal"  # Basic text extraction only
    STANDARD = "standard"  # Text + structure extraction
    COMPREHENSIVE = "comprehensive"  # Full multimodal with OCR and patterns


@dataclass
class MultimodalConfig:
    """Configuration for multimodal processing features."""

    # Feature flags
    enable_code_analysis: bool = True
    enable_data_extraction: bool = True
    enable_image_processing: bool = False  # Requires optional deps
    enable_pattern_detection: bool = True
    enable_cross_modal_linking: bool = True

    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.STANDARD
    max_file_size_mb: int = 50
    max_files_per_batch: int = 100
    chunk_overlap_ratio: float = 0.1

    # Code analysis settings
    code_languages: List[str] = field(
        default_factory=lambda: ["python", "javascript", "typescript"]
    )
    extract_docstrings: bool = True
    extract_comments: bool = True
    max_ast_depth: int = 10
    include_private_entities: bool = False

    # Data extraction settings
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    json_max_depth: int = 5
    extract_schema: bool = True
    infer_data_types: bool = True

    # Image processing settings (when enabled)
    ocr_language: str = "eng"
    max_image_dimension: int = 4096
    jpeg_quality: int = 85
    extract_text_from_images: bool = True
    extract_diagrams: bool = False

    # Pattern detection settings
    min_pattern_frequency: int = 3
    similarity_threshold: float = 0.8
    use_semantic_similarity: bool = True
    pattern_categories: List[str] = field(
        default_factory=lambda: [
            "api_patterns",
            "error_patterns",
            "design_patterns",
            "data_patterns",
        ]
    )

    # Graph building settings
    max_entities_per_file: int = 1000
    max_relationships_per_entity: int = 100
    merge_similar_entities: bool = True
    entity_similarity_threshold: float = 0.9

    # Performance settings
    use_caching: bool = True
    cache_ttl_seconds: int = 3600
    parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 10

    # Output settings
    include_embeddings: bool = True
    store_raw_content: bool = False
    compress_content: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "networkx"])

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "features": {
                "code_analysis": self.enable_code_analysis,
                "data_extraction": self.enable_data_extraction,
                "image_processing": self.enable_image_processing,
                "pattern_detection": self.enable_pattern_detection,
                "cross_modal_linking": self.enable_cross_modal_linking,
            },
            "processing": {
                "mode": self.processing_mode.value,
                "max_file_size_mb": self.max_file_size_mb,
                "max_files_per_batch": self.max_files_per_batch,
                "chunk_overlap_ratio": self.chunk_overlap_ratio,
            },
            "code_settings": {
                "languages": self.code_languages,
                "extract_docstrings": self.extract_docstrings,
                "extract_comments": self.extract_comments,
                "max_ast_depth": self.max_ast_depth,
                "include_private": self.include_private_entities,
            },
            "data_settings": {
                "csv_delimiter": self.csv_delimiter,
                "csv_encoding": self.csv_encoding,
                "json_max_depth": self.json_max_depth,
                "extract_schema": self.extract_schema,
                "infer_types": self.infer_data_types,
            },
            "pattern_settings": {
                "min_frequency": self.min_pattern_frequency,
                "similarity_threshold": self.similarity_threshold,
                "use_semantic": self.use_semantic_similarity,
                "categories": self.pattern_categories,
            },
            "graph_settings": {
                "max_entities_per_file": self.max_entities_per_file,
                "max_relationships_per_entity": self.max_relationships_per_entity,
                "merge_similar": self.merge_similar_entities,
                "similarity_threshold": self.entity_similarity_threshold,
            },
            "performance": {
                "caching": self.use_caching,
                "cache_ttl": self.cache_ttl_seconds,
                "parallel": self.parallel_processing,
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
            },
        }

    @classmethod
    def minimal(cls) -> "MultimodalConfig":
        """Create a minimal configuration for basic processing."""
        return cls(
            processing_mode=ProcessingMode.MINIMAL,
            enable_image_processing=False,
            enable_pattern_detection=False,
            enable_cross_modal_linking=False,
            parallel_processing=False,
            max_workers=1,
        )

    @classmethod
    def comprehensive(cls) -> "MultimodalConfig":
        """Create a comprehensive configuration with all features."""
        return cls(
            processing_mode=ProcessingMode.COMPREHENSIVE,
            enable_image_processing=True,
            enable_pattern_detection=True,
            enable_cross_modal_linking=True,
            extract_comments=True,
            extract_diagrams=True,
            include_private_entities=True,
            parallel_processing=True,
            max_workers=8,
        )

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.

        Returns:
            List of warning messages (empty if valid)
        """
        warnings = []

        # Check dependencies for image processing
        if self.enable_image_processing:
            try:
                import PIL  # noqa: F401
                import pytesseract  # noqa: F401
            except ImportError:
                warnings.append(
                    "Image processing enabled but PIL/pytesseract not installed. "
                    "Install with: pip install pillow pytesseract"
                )

        # Check data extraction dependencies
        if self.enable_data_extraction:
            try:
                import pandas  # noqa: F401
            except ImportError:
                warnings.append(
                    "Data extraction enabled but pandas not installed. "
                    "Install with: pip install pandas"
                )

        # Validate thresholds
        if not 0 <= self.similarity_threshold <= 1:
            warnings.append(
                f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}"
            )

        if not 0 <= self.entity_similarity_threshold <= 1:
            warnings.append(
                f"entity_similarity_threshold must be between 0 and 1, "
                f"got {self.entity_similarity_threshold}"
            )

        # Check performance settings
        if self.parallel_processing and self.max_workers < 1:
            warnings.append(
                f"max_workers must be >= 1 for parallel processing, got {self.max_workers}"
            )

        if self.batch_size < 1:
            warnings.append(f"batch_size must be >= 1, got {self.batch_size}")

        return warnings
