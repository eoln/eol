# Advanced RAG Patterns with Redis Implementation

## Overview

Advanced Retrieval-Augmented Generation (RAG) techniques that leverage Redis v8 for high-performance, real-time AI applications.

## Core RAG Techniques

### 1. GraphRAG

#### Concept

- Converts data into structured knowledge graphs
- Queries structured as graph queries for efficient retrieval
- LLMs reason over underlying graphs for contextually rich responses

#### Implementation Approaches

- **Microsoft GraphRAG**: Extracts entities and relationships, generates community summaries bottom-up
- **LightRAG**: Simplified version without community structure
- **LazyGraphRAG**: Uses smaller local models, dynamic query-time processing

#### Redis Implementation

```python
from redisvl import SearchIndex
import networkx as nx

class GraphRAG:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.graph = nx.Graph()

    def build_knowledge_graph(self, documents):
        """Extract entities and build graph structure"""
        for doc in documents:
            entities = self.extract_entities(doc)
            relationships = self.extract_relationships(entities)

            # Store in Redis as graph structure
            for entity in entities:
                self.redis.hset(f"entity:{entity.id}", mapping={
                    "name": entity.name,
                    "type": entity.type,
                    "embedding": entity.embedding.tobytes()
                })

            # Store relationships
            for rel in relationships:
                self.redis.sadd(f"rel:{rel.source}:{rel.type}", rel.target)

    def query_graph(self, query, k=5):
        """Query knowledge graph with semantic search"""
        query_embedding = self.embed(query)

        # Find relevant entities via vector search
        entities = self.vector_search(query_embedding, k)

        # Traverse graph for context
        context = self.traverse_relationships(entities)
        return context
```

### 2. HyDE (Hypothetical Document Embeddings)

#### Concept

- Generates hypothetical documents based on queries
- Enriches queries with inferred context
- Particularly useful for vague or under-specified queries

#### Redis Implementation

```python
class HyDERetriever:
    def __init__(self, redis_client, llm, embedder):
        self.redis = redis_client
        self.llm = llm
        self.embedder = embedder

    def retrieve(self, query):
        # Generate hypothetical document
        hypothetical = self.llm.generate(
            f"Write a detailed answer to: {query}"
        )

        # Embed hypothetical document
        hyde_embedding = self.embedder.encode(hypothetical)

        # Search Redis for similar documents
        results = self.redis.ft().search(
            Query("*=>[KNN 10 @embedding $vec AS score]")
            .dialect(2)
            .sort_by("score")
            .paging(0, 10),
            query_params={"vec": hyde_embedding.tobytes()}
        )

        return results
```

#### HyPE (Hypothetical Prompt Embeddings)

- Precomputes hypothetical prompts at indexing stage
- Transforms retrieval into question-question matching
- No LLM calls at query time

```python
class HyPEIndexer:
    def precompute_prompts(self, document):
        """Generate hypothetical questions for document"""
        prompts = self.llm.generate(
            f"Generate 5 questions that could be answered by: {document}"
        )

        for prompt in prompts:
            embedding = self.embedder.encode(prompt)
            self.redis.hset(f"hype:{document.id}:{prompt.id}", {
                "question": prompt,
                "doc_id": document.id,
                "embedding": embedding.tobytes()
            })
```

### 3. Self-RAG

#### Concept

- LLM evaluates its own outputs
- Decides whether to retrieve additional information
- Multi-step process with self-assessment

#### Implementation

```python
class SelfRAG:
    def __init__(self, redis_client, llm, retriever):
        self.redis = redis_client
        self.llm = llm
        self.retriever = retriever

    def generate_with_self_assessment(self, query):
        # Step 1: Decide if retrieval needed
        needs_retrieval = self.llm.evaluate(
            f"Does this query need external information: {query}?"
        )

        if needs_retrieval:
            # Step 2: Retrieve documents
            docs = self.retriever.search(query)

            # Step 3: Evaluate relevance
            relevant_docs = []
            for doc in docs:
                relevance = self.llm.evaluate(
                    f"Is this relevant to '{query}': {doc.content}"
                )
                if relevance > 0.7:
                    relevant_docs.append(doc)

            # Step 4: Generate response
            response = self.llm.generate(query, context=relevant_docs)

            # Step 5: Assess support
            support_score = self.llm.evaluate(
                f"How well supported is this response: {response}"
            )

            # Step 6: Utility evaluation
            if support_score < 0.5:
                return self.generate_with_self_assessment(query)

            return response

        return self.llm.generate(query)
```

### 4. Corrective RAG (CRAG)

#### Concept

- Scores and filters retrieved documents
- Lightweight retrieval evaluator
- Confidence-based knowledge retrieval actions

#### Redis Implementation

```python
class CorrectiveRAG:
    def __init__(self, redis_client, evaluator):
        self.redis = redis_client
        self.evaluator = evaluator

    def retrieve_and_correct(self, query):
        # Initial retrieval
        docs = self.vector_search(query)

        # Evaluate retrieval quality
        evaluations = []
        for doc in docs:
            score = self.evaluator.assess(query, doc)
            evaluations.append({
                "doc": doc,
                "score": score,
                "status": self.classify_score(score)
            })

        # Handle based on status
        correct_docs = [e["doc"] for e in evaluations
                       if e["status"] == "Correct"]

        if len(correct_docs) < 3:
            # Fallback to web search or alternative sources
            additional = self.web_search(query)
            correct_docs.extend(additional)

        return correct_docs

    def classify_score(self, score):
        if score > 0.8:
            return "Correct"
        elif score > 0.5:
            return "Ambiguous"
        else:
            return "Incorrect"
```

### 5. HybridRAG

#### Concept

- Combines GraphRAG with VectorRAG
- Addresses complex relationship queries
- Provides global context from entire datasets

#### Implementation

```python
class HybridRAG:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.vector_index = SearchIndex(redis_client)
        self.graph_index = GraphRAG(redis_client)

    def retrieve(self, query, vector_weight=0.5):
        # Vector-based retrieval
        vector_results = self.vector_index.search(
            query,
            k=10,
            return_fields=["content", "metadata"]
        )

        # Graph-based retrieval
        graph_results = self.graph_index.query_graph(
            query,
            k=10
        )

        # Combine results with weighting
        combined = self.merge_results(
            vector_results,
            graph_results,
            vector_weight
        )

        return combined

    def merge_results(self, vector_res, graph_res, weight):
        """Intelligent merging of vector and graph results"""
        merged = {}

        # Score and deduplicate
        for res in vector_res:
            merged[res.id] = {
                "content": res.content,
                "score": res.score * weight
            }

        for res in graph_res:
            if res.id in merged:
                merged[res.id]["score"] += res.score * (1 - weight)
            else:
                merged[res.id] = {
                    "content": res.content,
                    "score": res.score * (1 - weight)
                }

        # Sort by combined score
        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)
```

## Redis-Specific RAG Optimizations

### 1. Semantic Caching

```python
class SemanticCache:
    def __init__(self, redis_client, similarity_threshold=0.97):
        self.redis = redis_client
        self.threshold = similarity_threshold

    def get_or_generate(self, query, generator):
        # Check cache
        query_embedding = self.embed(query)

        cached = self.redis.ft().search(
            Query("*=>[KNN 1 @query_embedding $vec AS similarity]")
            .add_filter(f"@similarity >= {self.threshold}")
            .dialect(2),
            query_params={"vec": query_embedding.tobytes()}
        )

        if cached.docs:
            # Cache hit (31% of queries)
            return cached.docs[0].response

        # Cache miss - generate and store
        response = generator(query)

        self.redis.hset(f"cache:{hash(query)}", {
            "query": query,
            "query_embedding": query_embedding.tobytes(),
            "response": response,
            "timestamp": time.time()
        })

        return response
```

### 2. Chunking Strategies

```python
class SmartChunker:
    def __init__(self, redis_client):
        self.redis = redis_client

    def semantic_chunk(self, text, max_tokens=512):
        """Semantic chunking based on topic coherence"""
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            if current_tokens + sent_tokens > max_tokens:
                # Check semantic coherence
                if self.is_coherent(current_chunk, sent):
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    # Split at semantic boundary
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def store_chunks(self, doc_id, chunks):
        """Store chunks with metadata in Redis"""
        for i, chunk in enumerate(chunks):
            embedding = self.embed(chunk)

            self.redis.hset(f"chunk:{doc_id}:{i}", {
                "content": chunk,
                "embedding": embedding.tobytes(),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
```

### 3. Real-time Context Updates

```python
class RealtimeRAG:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.pubsub = redis_client.pubsub()

    def update_context(self, doc_id, new_content):
        """Real-time context updates with pub/sub"""
        # Update document
        embedding = self.embed(new_content)

        self.redis.hset(f"doc:{doc_id}", {
            "content": new_content,
            "embedding": embedding.tobytes(),
            "updated_at": time.time()
        })

        # Notify subscribers
        self.redis.publish("context_updates", json.dumps({
            "doc_id": doc_id,
            "action": "update"
        }))

    def subscribe_to_updates(self, callback):
        """Subscribe to real-time context changes"""
        self.pubsub.subscribe("context_updates")

        for message in self.pubsub.listen():
            if message["type"] == "message":
                update = json.loads(message["data"])
                callback(update)
```

## Performance Metrics

### Caching Impact

- **31% cache hit rate** for semantic queries
- **Response time**: Seconds (LLM) vs milliseconds (cache)
- **Cost reduction**: Up to 30% fewer LLM API calls

### Hybrid Approach Performance

- **Context Recall**: 1.0 (matching VectorRAG)
- **Accuracy**: Outperforms pure GraphRAG (0.85)
- **Latency**: Sub-100ms for cached queries

## EOL Framework Integration

### Prototyping Phase

```yaml
# rag-feature.eol
name: advanced-rag-system
phase: prototyping

rag_config:
  technique: hybrid
  components:
    - graphrag
    - vectorrag
    - semantic_cache

operations:
  - "Build knowledge graph from documents"
  - "Index documents with semantic chunking"
  - "Enable semantic caching with 97% similarity"
  - "Query with hybrid retrieval"
```

### Implementation Phase

```python
# Generated from rag-feature.eol
class EOLAdvancedRAG:
    def __init__(self, redis_config):
        self.redis = Redis(**redis_config)
        self.hybrid_rag = HybridRAG(self.redis)
        self.cache = SemanticCache(self.redis)
        self.chunker = SmartChunker(self.redis)

    async def process_query(self, query):
        # Check cache first
        cached = await self.cache.get_or_generate(
            query,
            lambda q: self.hybrid_rag.retrieve(q)
        )

        if cached:
            return cached

        # Perform hybrid retrieval
        results = await self.hybrid_rag.retrieve(query)

        # Generate response
        response = await self.generate_response(query, results)

        # Cache for future
        await self.cache.store(query, response)

        return response
```

## Best Practices

1. **Choose RAG technique based on use case**:
   - GraphRAG for relationship-heavy queries
   - HyDE for vague queries
   - Self-RAG for accuracy-critical applications
   - CRAG for quality assurance

2. **Implement semantic caching** for:
   - FAQs and documentation
   - Stable knowledge bases
   - High-traffic queries

3. **Optimize chunking** based on:
   - Document structure
   - Query patterns
   - Token limits

4. **Monitor and iterate**:
   - Track cache hit rates
   - Measure retrieval relevance
   - Adjust similarity thresholds
