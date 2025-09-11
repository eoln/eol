# Temporal Context Preservation Update

## What Was Fixed

The XML processing system has been enhanced to properly preserve temporal (date/time) information when processing event XML files. Previously, date information was being separated from event details during chunking, leading to search results without temporal context.

## Key Improvements

### 1. Event XML Detection

- System now automatically detects event XML files (with `<event>` root tag)
- Special processing for theater, concert, and cultural event data

### 2. Single-Chunk Strategy for Events

- Event XMLs are now processed as single, comprehensive chunks
- All temporal information (date, time, location) stays together with event details
- This ensures date/time context is never lost during search

### 3. Enhanced Temporal Extraction

- Searches for date/time information in multiple locations:
  - `<calendar><date>` elements
  - Direct `<date>` elements
  - Temporal attributes in any element
- Preserves raw date strings (e.g., "7 czerwca 12:00")

### 4. Structured Content Format

Each event chunk now contains:

```
Event: [Event Title]
Date/Time: [Date and Time]
Location: [Venue and Address]
Event ID: [Unique ID]

Details:
[Full event description]

More info: [URL if available]
```

## How to Use

### Indexing Event XMLs

```python
# The system automatically detects and properly processes event XMLs
await indexer.index_folder("/path/to/event/xmls")
```

### Searching with Temporal Context

When searching, the temporal information is included in both:

1. **Content**: Date/time is part of the searchable text
2. **Metadata**: Date is stored in chunk metadata for filtering

Example searches that now work correctly:

- "wydarzenia w czerwcu" (events in June)
- "teatr 7 czerwca" (theater on June 7th)
- "koncerty wieczorem" (evening concerts)
- "wydarzenia w weekend" (weekend events)

## Testing

Run the test scripts to verify temporal context preservation:

```bash
# Test individual file processing
python test_event_xml.py

# Test full indexing and search
python test_event_indexing.py
```

## Implementation Details

### Modified Files

1. **document_processor.py**:
   - Added `_chunk_xml_event()` method for event-specific processing
   - Enhanced `_chunk_xml_generic()` with temporal context inheritance
   - Added `_extract_temporal_metadata()` helper method

2. **config.py**:
   - XML file patterns already included (*.xml)

3. **mcp_launcher_final.py**:
   - No changes needed - already passes content correctly

### Data Model

Event chunks include these metadata fields:

- `date`: Human-readable date/time string
- `datetime_raw`: Original date string from XML
- `location`: Venue and address
- `event_id`: Unique event identifier
- `title`: Event title
- `url`: Link to more information

## Benefits

1. **Complete Context**: Users get full temporal information with search results
2. **Better Search Relevance**: Dates are searchable and part of the content
3. **Preserved Structure**: Event information stays logically grouped
4. **Metadata Filtering**: Can filter results by date in metadata

## Notes for Polish Date Formats

The system preserves Polish date formats as-is:

- "7 czerwca 12:00" (June 7th at 12:00)
- "5 czerwca 19:00" (June 5th at 19:00)

These are searchable in Polish and the system maintains the original formatting.
