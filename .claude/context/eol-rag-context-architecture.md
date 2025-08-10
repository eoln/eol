# EOL-RAG-Context MCP Server Architecture

## Overview
The `eol-rag-context` is an intelligent context management MCP server that replaces static `.claude/context` files with a dynamic, Redis 8-backed RAG system. It provides optimal context structuring for LLMs based on 2024 best practices.

## Core Principles

### 1. Hierarchical Context Organization
Based on research showing that hierarchical processing is crucial for managing extensive context:
- **Level 1: Concepts** - High-level semantic abstractions
- **Level 2: Sections** - Grouped related content
- **Level 3: Chunks** - Fine-grained, retrievable units
- **Level 4: Tokens** - Raw content for exact matching

### 2. Strategic Information Placement
Following the "lost in the middle" phenomenon:
- **Critical information** at the beginning (system instructions)
- **Retrieved context** in clearly labeled sections
- **User query** near the end for recency
- **Examples** strategically placed based on relevance

### 3. Dynamic Context Windows
Implementing HOMER (Hierarchical Context Merging) approach:
- Divide long inputs into manageable chunks
- Progressive merging at transformer layers
- Adaptive context size based on query complexity

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  eol-rag-context MCP Server             │
├─────────────────────────────────────────────────────────┤
│                    MCP Interface Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Resources  │  │    Tools     │  │   Prompts    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Context Engine Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Hierarchical│  │   Semantic   │  │   Dynamic    │ │
│  │   Organizer  │  │   Analyzer   │  │   Composer   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                     RAG Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Indexer    │  │  Retriever   │  │  Augmenter   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Redis 8 Backend                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │Vector Store  │  │ Semantic Cache│  │  Metadata   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Optimal Context Structure

### 1. System Context Template
```python
@dataclass
class OptimalContext:
    """Optimal LLM context structure based on 2024 research"""
    
    # Priority 1: System instructions (beginning)
    system_role: str = field(default="")
    
    # Priority 2: Task-specific guidelines
    task_context: str = field(default="")
    
    # Priority 3: Retrieved context (clearly labeled)
    retrieved_sections: List[ContextSection] = field(default_factory=list)
    
    # Priority 4: Examples (few-shot)
    examples: List[Example] = field(default_factory=list)
    
    # Priority 5: Current conversation
    conversation_history: List[Message] = field(default_factory=list)
    
    # Priority 6: User query (end for recency)
    user_query: str = field(default="")

@dataclass
class ContextSection:
    """Hierarchical context section"""
    label: str  # Clear labeling e.g., "Retrieved Documentation:"
    relevance_score: float
    content: str
    metadata: Dict[str, Any]
    hierarchy_level: int  # 1=concept, 2=section, 3=chunk
```

### 2. Context Composition Strategy
```python
class ContextComposer:
    """Composes optimal context following 2024 best practices"""
    
    def compose_context(
        self,
        query: str,
        max_tokens: int = 32000,
        strategy: str = "hierarchical"
    ) -> OptimalContext:
        """
        Compose context using research-backed strategies:
        - Place critical info at beginning/end
        - Use hierarchical organization
        - Apply semantic grouping
        - Implement quality over quantity
        """
        
        context = OptimalContext()
        
        # 1. System instructions (always first)
        context.system_role = self._get_system_instructions()
        
        # 2. Task-specific context
        context.task_context = self._analyze_task(query)
        
        # 3. Retrieve relevant sections
        retrieved = self._retrieve_hierarchical(query)
        
        # 4. Apply strategic placement
        context.retrieved_sections = self._strategically_place(
            retrieved,
            max_tokens=max_tokens * 0.6  # Leave room for other sections
        )
        
        # 5. Add few-shot examples if helpful
        if self._needs_examples(query):
            context.examples = self._get_relevant_examples(query)
        
        # 6. Include conversation history (recent only)
        context.conversation_history = self._get_recent_history()
        
        # 7. Place user query at end
        context.user_query = query
        
        return context
```

## RAG Pipeline Implementation

### 1. Intelligent Indexing
```python
class HierarchicalIndexer:
    """Index content at multiple hierarchy levels"""
    
    async def index_document(self, doc_path: Path):
        """Index document with hierarchical structure"""
        
        content = await self._read_document(doc_path)
        
        # Level 1: Extract concepts (highest abstraction)
        concepts = await self._extract_concepts(content)
        
        # Level 2: Identify sections
        sections = self._segment_sections(content)
        
        # Level 3: Create semantic chunks
        chunks = []
        for section in sections:
            section_chunks = await self._chunk_semantically(
                section,
                chunk_size=512,  # Optimal based on research
                overlap=64
            )
            chunks.extend(section_chunks)
        
        # Store in Redis with hierarchy metadata
        await self._store_hierarchical(concepts, sections, chunks)
```

### 2. Multi-Level Retrieval
```python
class HierarchicalRetriever:
    """Retrieve context at appropriate hierarchy levels"""
    
    async def retrieve(
        self,
        query: str,
        max_chunks: int = 10,
        strategy: str = "adaptive"
    ) -> List[ContextSection]:
        """
        Retrieve using hierarchy:
        - Start with concept matching
        - Drill down to relevant sections
        - Return specific chunks
        """
        
        # Embed query
        query_embedding = await self._embed(query)
        
        # 1. Find relevant concepts
        concepts = await self._search_concepts(query_embedding, k=3)
        
        # 2. Find sections within concepts
        sections = []
        for concept in concepts:
            concept_sections = await self._search_sections(
                query_embedding,
                concept_filter=concept.id,
                k=5
            )
            sections.extend(concept_sections)
        
        # 3. Get specific chunks if needed
        if self._needs_detail(query):
            chunks = await self._search_chunks(
                query_embedding,
                section_filters=[s.id for s in sections],
                k=max_chunks
            )
            sections.extend(chunks)
        
        # 4. Apply quality filtering
        return self._filter_by_quality(sections)
```

### 3. Context Augmentation
```python
class ContextAugmenter:
    """Augment retrieved context for optimal LLM consumption"""
    
    def augment_for_llm(
        self,
        sections: List[ContextSection],
        query: str
    ) -> str:
        """Format context following 2024 best practices"""
        
        augmented = []
        
        # Group by hierarchy level
        concepts = [s for s in sections if s.hierarchy_level == 1]
        mid_sections = [s for s in sections if s.hierarchy_level == 2]
        details = [s for s in sections if s.hierarchy_level == 3]
        
        # Add high-level concepts first
        if concepts:
            augmented.append("## Key Concepts:")
            for concept in concepts[:2]:  # Limit to avoid overload
                augmented.append(f"- {concept.content}")
        
        # Add relevant sections with clear labels
        if mid_sections:
            augmented.append("\n## Retrieved Context:")
            for section in mid_sections[:3]:
                augmented.append(f"\n### {section.label}")
                augmented.append(section.content)
        
        # Add specific details if needed
        if details and self._query_needs_detail(query):
            augmented.append("\n## Specific Information:")
            for detail in details[:5]:
                augmented.append(f"- {detail.content}")
        
        return "\n".join(augmented)
```

## MCP Server Implementation

### MCP Resources
```python
@mcp.resource("context://query/{query}")
async def get_context(query: str) -> Dict:
    """Get optimized context for a query"""
    
    context = await context_composer.compose_context(query)
    return context.to_dict()

@mcp.resource("context://hierarchy/{level}")
async def get_hierarchy_level(level: int) -> List[Dict]:
    """Get all items at a specific hierarchy level"""
    
    return await hierarchy_store.get_level(level)
```

### MCP Tools
```python
@mcp.tool()
async def index_directory(
    path: str,
    file_patterns: List[str] = ["*.md", "*.py", "*.yaml"]
) -> Dict:
    """Index a directory of files into the RAG system"""
    
    indexed_count = 0
    for pattern in file_patterns:
        files = Path(path).glob(f"**/{pattern}")
        for file in files:
            await indexer.index_document(file)
            indexed_count += 1
    
    return {"indexed": indexed_count}

@mcp.tool()
async def search_context(
    query: str,
    max_results: int = 10,
    min_relevance: float = 0.7
) -> List[Dict]:
    """Search for relevant context"""
    
    results = await retriever.retrieve(query, max_results)
    return [r.to_dict() for r in results if r.relevance_score >= min_relevance]

@mcp.tool()
async def optimize_context(
    current_context: str,
    target_tokens: int = 32000
) -> str:
    """Optimize context size and structure"""
    
    return await optimizer.optimize(current_context, target_tokens)
```

### MCP Prompts
```python
@mcp.prompt("structured_query")
async def structured_query_prompt() -> str:
    """Generate a structured query for RAG retrieval"""
    
    return """
    Transform the user query into a structured format:
    
    1. Main Intent: [What is the user trying to achieve?]
    2. Key Entities: [Important nouns/concepts]
    3. Context Level: [concept/section/detail]
    4. Required Depth: [shallow/medium/deep]
    5. Output Format: [explanation/code/list/comparison]
    """

@mcp.prompt("context_synthesis")
async def context_synthesis_prompt() -> str:
    """Synthesize multiple context sections"""
    
    return """
    Given the retrieved context sections, synthesize them by:
    
    1. Identifying common themes
    2. Resolving contradictions
    3. Hierarchically organizing information
    4. Highlighting most relevant parts
    5. Summarizing if exceeds token limit
    """
```

## Redis 8 Schema

### Vector Indexes
```yaml
# Concept-level index
concept_index:
  type: HNSW
  dimension: 1536
  distance_metric: COSINE
  initial_cap: 1000
  m: 16
  ef_construction: 200

# Section-level index  
section_index:
  type: HNSW
  dimension: 1536
  distance_metric: COSINE
  initial_cap: 10000
  m: 24
  ef_construction: 300

# Chunk-level index
chunk_index:
  type: FLAT  # For exact search on smaller sets
  dimension: 1536
  distance_metric: COSINE
  initial_cap: 100000
```

### Data Structure
```python
# Hierarchical storage in Redis
{
    # Concept level
    "concept:{id}": {
        "content": "High-level concept description",
        "embedding": [...],
        "children": ["section:1", "section:2"],
        "metadata": {...}
    },
    
    # Section level
    "section:{id}": {
        "content": "Section content",
        "embedding": [...],
        "parent": "concept:1",
        "children": ["chunk:1", "chunk:2"],
        "metadata": {...}
    },
    
    # Chunk level
    "chunk:{id}": {
        "content": "Detailed chunk",
        "embedding": [...],
        "parent": "section:1",
        "metadata": {...}
    }
}
```

## Performance Optimizations

### 1. Semantic Caching
- Cache frequently accessed context combinations
- 31% hit rate target using similarity threshold
- TTL based on access patterns

### 2. Progressive Loading
- Start with concepts for quick response
- Load sections/chunks as needed
- Stream context to LLM progressively

### 3. Quality Filtering
- Relevance scoring at each level
- Remove redundant information
- Prioritize recent and high-quality sources

## Integration with EOL Framework

The `eol-rag-context` server integrates with EOL as a core dependency:

```yaml
# In .eol.md files
dependencies:
  mcp_servers:
    - name: eol-rag-context
      version: ">=1.0.0"
      config:
        redis_url: ${REDIS_URL}
        max_context_tokens: 32000
        hierarchy_levels: 3
```

This architecture provides intelligent, dynamic context management that adapts to each query's needs while following 2024's best practices for LLM context optimization.