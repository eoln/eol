# EOL SDK

**Status**: 📋 Planned  
**Target Release**: Q2 2025

## Overview

EOL SDK will provide high-level Python APIs for building RAG-powered applications. It will abstract away complexity while maintaining flexibility for advanced use cases.

## Planned Features

### High-Level APIs
- Simple interfaces for common tasks
- Chainable operations
- Async and sync support
- Type-safe throughout

### Application Templates
- Chat applications
- Document Q&A systems
- Code assistants
- Knowledge bases

### Integration Helpers
- LangChain integration
- LlamaIndex compatibility
- FastAPI utilities
- Streamlit components

### Advanced Features
- Custom pipelines
- Plugin system
- Middleware support
- Event hooks

## Example Usage (Planned)

```python
from eol_sdk import RAGApplication

# Create application
app = RAGApplication("my-app")

# Index documents
app.index_documents("./docs")

# Simple Q&A
answer = app.ask("What is the main feature?")

# Conversational interface
chat = app.create_chat()
response = chat.send("Hello!")

# Advanced pipeline
pipeline = (
    app.pipeline()
    .add_preprocessor(custom_cleaner)
    .add_chunker(semantic_chunker)
    .add_embedder(openai_embedder)
    .add_retriever(hybrid_retriever)
    .build()
)

results = pipeline.run("complex query")
```

## Design Principles

- **Developer-First**: Intuitive APIs that feel Pythonic
- **Progressive Disclosure**: Simple by default, powerful when needed
- **Type-Safe**: Full type hints for better IDE support
- **Well-Documented**: Examples for every feature
- **Testable**: Built with testing in mind

## Use Cases

### Document Q&A System
```python
from eol_sdk import DocumentQA

qa = DocumentQA()
qa.add_documents("./knowledge-base")
answer = qa.answer("What is the process?")
```

### Conversational AI
```python
from eol_sdk import ConversationalRAG

bot = ConversationalRAG()
bot.set_context("./product-docs")
response = bot.chat("Tell me about pricing")
```

### Code Assistant
```python
from eol_sdk import CodeAssistant

assistant = CodeAssistant()
assistant.index_repository("./src")
help = assistant.explain("How does authentication work?")
```

## Roadmap

- [ ] API design document
- [ ] Core interfaces
- [ ] Application templates
- [ ] Integration modules
- [ ] Example applications
- [ ] Documentation
- [ ] Beta release
- [ ] Stable release

## Contributing

Help us design the SDK:

1. Share use cases in [Discussions](https://github.com/eoln/eol/discussions)
2. Review [API proposals](https://github.com/eoln/eol/issues?q=is%3Aopen+is%3Aissue+label%3Aeol-sdk)
3. Suggest features and improvements

---

This package is in the design phase. We'd love your input on the API design!