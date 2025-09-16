"""
Data extraction module for CSV, JSON, and structured data files.
Extracts entities, relationships, and schemas from data files.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DataEntity:
    """Represents an entity extracted from data files."""

    id: str
    type: str
    name: str
    content: str
    metadata: Dict[str, Any]
    schema: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "content": self.content,
            "metadata": self.metadata,
            "schema": self.schema,
        }


@dataclass
class DataRelation:
    """Represents a relationship between data entities."""

    source: str
    target: str
    type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "metadata": self.metadata,
        }


class DataExtractor:
    """Extracts entities and relationships from structured data files."""

    def __init__(self, config):
        """
        Initialize data extractor.

        Args:
            config: Multimodal configuration
        """
        self.config = config
        self._pandas_available = False

        # Check if pandas is available for advanced data processing
        try:
            import pandas as pd

            self._pandas_available = True
            self.pd = pd
        except ImportError:
            logger.warning("Pandas not available. Using basic CSV/JSON parsing.")

    async def extract_from_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from a data file.

        Args:
            file_path: Path to the data file

        Returns:
            Tuple of (entities, relationships) as dictionaries
        """
        entities = []
        relationships = []

        try:
            suffix = file_path.suffix.lower()

            if suffix == ".json":
                entities, relationships = await self._extract_from_json(file_path)
            elif suffix == ".csv":
                entities, relationships = await self._extract_from_csv(file_path)
            elif suffix in [".jsonl", ".ndjson"]:
                entities, relationships = await self._extract_from_jsonlines(file_path)
            elif suffix in [".xml"]:
                entities, relationships = await self._extract_from_xml(file_path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")

        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")

        return entities, relationships

    async def _extract_from_json(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from JSON file."""
        entities = []
        relationships = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract schema if configured
            schema = None
            if self.config.extract_schema:
                schema = self._extract_json_schema(data)

            # Create entity for the JSON file itself
            file_entity = DataEntity(
                id=f"json_{file_path.stem}",
                type="json_file",
                name=file_path.name,
                content=json.dumps(data, indent=2)[:1000],  # Limited content
                metadata={
                    "file_path": str(file_path),
                    "size": file_path.stat().st_size,
                    "root_type": type(data).__name__,
                },
                schema=schema,
            )
            entities.append(file_entity.to_dict())

            # Extract nested entities based on structure
            if isinstance(data, dict):
                nested_entities, nested_relations = self._extract_from_dict(
                    data, file_entity.id, file_path.stem
                )
                entities.extend(nested_entities)
                relationships.extend(nested_relations)
            elif isinstance(data, list):
                nested_entities, nested_relations = self._extract_from_list(
                    data, file_entity.id, file_path.stem
                )
                entities.extend(nested_entities)
                relationships.extend(nested_relations)

        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")

        return entities, relationships

    async def _extract_from_csv(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from CSV file."""
        entities = []
        relationships = []

        try:
            if self._pandas_available:
                # Use pandas for advanced processing
                df = self.pd.read_csv(
                    file_path,
                    delimiter=self.config.csv_delimiter,
                    encoding=self.config.csv_encoding,
                )

                # Create entity for the CSV file
                file_entity = DataEntity(
                    id=f"csv_{file_path.stem}",
                    type="csv_file",
                    name=file_path.name,
                    content=f"CSV with {len(df)} rows and {len(df.columns)} columns",
                    metadata={
                        "file_path": str(file_path),
                        "rows": len(df),
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.to_dict() if self.config.infer_data_types else {},
                    },
                    schema=(
                        {"columns": list(df.columns), "shape": df.shape}
                        if self.config.extract_schema
                        else None
                    ),
                )
                entities.append(file_entity.to_dict())

                # Extract column entities
                for col in df.columns:
                    col_entity = DataEntity(
                        id=f"csv_column_{file_path.stem}_{col}",
                        type="csv_column",
                        name=col,
                        content=f"Column {col} with {df[col].nunique()} unique values",
                        metadata={
                            "dtype": str(df[col].dtype),
                            "unique_values": df[col].nunique(),
                            "null_count": df[col].isnull().sum(),
                            "sample_values": (
                                df[col].dropna().head(5).tolist() if len(df[col]) > 0 else []
                            ),
                        },
                    )
                    entities.append(col_entity.to_dict())

                    # Create relationship between file and column
                    relationships.append(
                        DataRelation(
                            source=file_entity.id,
                            target=col_entity.id,
                            type="has_column",
                            metadata={"position": list(df.columns).index(col)},
                        ).to_dict()
                    )

                # Detect relationships between columns
                if self.config.infer_data_types:
                    column_relations = self._detect_column_relationships(df)
                    relationships.extend(column_relations)

            else:
                # Basic CSV processing without pandas
                with open(file_path, "r", encoding=self.config.csv_encoding) as f:
                    reader = csv.DictReader(f, delimiter=self.config.csv_delimiter)
                    rows = list(reader)

                # Create entity for the CSV file
                file_entity = DataEntity(
                    id=f"csv_{file_path.stem}",
                    type="csv_file",
                    name=file_path.name,
                    content=f"CSV with {len(rows)} rows",
                    metadata={
                        "file_path": str(file_path),
                        "rows": len(rows),
                        "columns": list(rows[0].keys()) if rows else [],
                    },
                )
                entities.append(file_entity.to_dict())

        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")

        return entities, relationships

    async def _extract_from_jsonlines(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from JSONL/NDJSON files."""
        entities = []
        relationships = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Create entity for the JSONL file
            file_entity = DataEntity(
                id=f"jsonl_{file_path.stem}",
                type="jsonl_file",
                name=file_path.name,
                content=f"JSONL with {len(lines)} records",
                metadata={"file_path": str(file_path), "record_count": len(lines)},
            )
            entities.append(file_entity.to_dict())

            # Sample first few records for schema extraction
            sample_records = []
            for i, line in enumerate(lines[:10]):  # Sample first 10
                try:
                    record = json.loads(line.strip())
                    sample_records.append(record)
                except json.JSONDecodeError:
                    continue

            # Extract schema from samples
            if self.config.extract_schema and sample_records:
                schema = self._extract_json_schema(sample_records[0])
                file_entity.metadata["schema"] = schema

        except Exception as e:
            logger.error(f"Error processing JSONL file {file_path}: {e}")

        return entities, relationships

    async def _extract_from_xml(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from XML files."""
        entities = []
        relationships = []

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Create entity for XML file
            file_entity = DataEntity(
                id=f"xml_{file_path.stem}",
                type="xml_file",
                name=file_path.name,
                content=f"XML with root element: {root.tag}",
                metadata={
                    "file_path": str(file_path),
                    "root_tag": root.tag,
                    "namespaces": dict(root.attrib) if root.attrib else {},
                },
            )
            entities.append(file_entity.to_dict())

            # Extract element entities
            elements = self._extract_xml_elements(root, file_entity.id)
            entities.extend(elements)

        except Exception as e:
            logger.error(f"Error processing XML file {file_path}: {e}")

        return entities, relationships

    def _extract_from_dict(
        self, data: Dict, parent_id: str, prefix: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from dictionary structure."""
        entities = []
        relationships = []

        for key, value in data.items():
            if self.config.json_max_depth > 0:
                entity_id = f"{prefix}_{key}"
                entity = DataEntity(
                    id=entity_id,
                    type="json_object" if isinstance(value, dict) else "json_field",
                    name=key,
                    content=str(value)[:500],
                    metadata={"value_type": type(value).__name__, "parent": parent_id},
                )
                entities.append(entity.to_dict())

                # Create relationship to parent
                relationships.append(
                    DataRelation(
                        source=parent_id, target=entity_id, type="contains", metadata={"key": key}
                    ).to_dict()
                )

                # Recursively process nested structures
                if isinstance(value, dict):
                    nested_entities, nested_relations = self._extract_from_dict(
                        value, entity_id, f"{prefix}_{key}"
                    )
                    entities.extend(nested_entities)
                    relationships.extend(nested_relations)

        return entities, relationships

    def _extract_from_list(
        self, data: List, parent_id: str, prefix: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities from list structure."""
        entities = []
        relationships = []

        # Sample first few items for pattern detection
        sample_size = min(10, len(data))
        for i in range(sample_size):
            item = data[i]
            entity_id = f"{prefix}_item_{i}"

            entity = DataEntity(
                id=entity_id,
                type="json_array_item",
                name=f"Item {i}",
                content=str(item)[:500],
                metadata={"index": i, "value_type": type(item).__name__, "parent": parent_id},
            )
            entities.append(entity.to_dict())

            # Create relationship to parent
            relationships.append(
                DataRelation(
                    source=parent_id, target=entity_id, type="contains", metadata={"index": i}
                ).to_dict()
            )

        return entities, relationships

    def _extract_json_schema(self, data: Any, depth: int = 0) -> Dict:
        """Extract schema from JSON data."""
        if depth >= self.config.json_max_depth:
            return {"type": type(data).__name__}

        if isinstance(data, dict):
            schema = {"type": "object", "properties": {}}
            for key, value in data.items():
                schema["properties"][key] = self._extract_json_schema(value, depth + 1)
            return schema
        elif isinstance(data, list):
            if data:
                return {"type": "array", "items": self._extract_json_schema(data[0], depth + 1)}
            else:
                return {"type": "array", "items": {}}
        else:
            return {"type": type(data).__name__}

    def _detect_column_relationships(self, df) -> List[Dict]:
        """Detect relationships between CSV columns using pandas."""
        relationships = []

        if not self._pandas_available:
            return relationships

        try:
            # Detect correlations for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Upper triangle only
                            correlation = corr_matrix.loc[col1, col2]
                            if abs(correlation) > self.config.similarity_threshold:
                                relationships.append(
                                    DataRelation(
                                        source=f"csv_column_{df.index.name or 'data'}_{col1}",
                                        target=f"csv_column_{df.index.name or 'data'}_{col2}",
                                        type="correlates_with",
                                        metadata={"correlation": float(correlation)},
                                    ).to_dict()
                                )

            # Detect potential foreign key relationships
            for col in df.columns:
                if col.endswith("_id") or col.endswith("Id"):
                    # This column might be a foreign key
                    base_name = col[:-3] if col.endswith("_id") else col[:-2]
                    for other_col in df.columns:
                        if other_col != col and base_name.lower() in other_col.lower():
                            relationships.append(
                                DataRelation(
                                    source=f"csv_column_{df.index.name or 'data'}_{col}",
                                    target=f"csv_column_{df.index.name or 'data'}_{other_col}",
                                    type="references",
                                    metadata={"key_type": "foreign_key"},
                                ).to_dict()
                            )

        except Exception as e:
            logger.warning(f"Error detecting column relationships: {e}")

        return relationships

    def _extract_xml_elements(self, element, parent_id: str, depth: int = 0) -> List[Dict]:
        """Recursively extract XML elements."""
        entities = []

        if depth >= self.config.json_max_depth:
            return entities

        for child in element:
            entity_id = f"xml_element_{child.tag}_{id(child)}"
            entity = DataEntity(
                id=entity_id,
                type="xml_element",
                name=child.tag,
                content=child.text[:500] if child.text else "",
                metadata={"attributes": dict(child.attrib), "parent": parent_id, "depth": depth},
            )
            entities.append(entity.to_dict())

            # Recursively process children
            entities.extend(self._extract_xml_elements(child, entity_id, depth + 1))

        return entities
