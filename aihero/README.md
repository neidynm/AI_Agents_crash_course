# FAQ Agent - Production Application 🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/)

This directory contains the production-ready implementation of the FAQ Agent, the final deliverable of the [AI Agents Email Crash-Course](https://alexeygrigorev.com/aihero/). The application is deployed on Streamlit Cloud and demonstrates a complete RAG (Retrieval-Augmented Generation) system.

## 🌐 Live Application

**Deployed URL**: [https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/](https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/)

## 📋 Overview

The FAQ Agent is an intelligent assistant that:
- ✅ Answers questions about Evidently AI documentation
- ✅ Uses hybrid search (text + semantic) for optimal retrieval
- ✅ Generates accurate, context-aware responses
- ✅ Provides source attribution for all answers
- ✅ Offers multiple search strategies

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│    Search System            │
│  ┌──────────────────────┐   │
│  │  Text Search (BM25)  │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │ Vector Search        │   │
│  │ (Sentence-BERT)      │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │ Hybrid Fusion        │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Context Building           │
│  (Top-5 Relevant Chunks)    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  LLM (DeepSeek via          │
│  OpenRouter)                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

## 📁 File Structure

```
src/
├── app.py              # Streamlit application (main entry point)
├── ingest.py           # Data ingestion from GitHub repos
├── search_tools.py     # Search implementations (text, vector, hybrid)
├── search_agent.py     # LLM integration and answer generation
├── logs.py             # Logging utilities (optional)
└── main.py             # CLI interface for testing
```

## 🔧 Core Components

### 1. `app.py` - Streamlit Application

The main user interface that orchestrates all components:

```python
# Key features:
- Chat-based interface
- Real-time search method selection
- Caching for performance (@st.cache_resource)
- Session state management for conversation history
- Example questions for quick start
```

**Run locally:**
```bash
streamlit run app.py
```

### 2. `ingest.py` - Data Ingestion

Handles downloading and processing GitHub repository content:

```python
# Main functions:
- read_repo_data(owner, repo, branch)  # Download repo as ZIP
- sliding_window(text, size, step)      # Create overlapping chunks
- ingest_repo(owner, repo)              # Full ingestion pipeline
- build_text_index(chunks)              # Build search index
```

**Features:**
- Downloads repositories from GitHub
- Parses Markdown/MDX files with frontmatter
- Creates overlapping text chunks (2000 chars, 1000 step)
- Builds searchable index

### 3. `search_tools.py` - Search Implementations

Three search strategies with seamless integration:

```python
# Classes & Functions:
- VectorSearch              # Cosine similarity search
- create_embeddings()       # Generate embeddings for corpus
- hybrid_search()           # Combine text + vector search
```

**Search Methods:**

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| Text | Keyword queries | ⚡⚡⚡ Fast | ⭐⭐ Good |
| Vector | Semantic queries | ⚡⚡ Moderate | ⭐⭐⭐ Excellent |
| Hybrid | General queries | ⚡⚡ Moderate | ⭐⭐⭐⭐ Best |

### 4. `search_agent.py` - LLM Integration

Connects search results with language model for answer generation:

```python
# Main function:
- answer_question_manual(question, search_method, index, vindex)
```

**Process:**
1. Search documentation (text/vector/hybrid)
2. Format top-5 results as context
3. Create structured prompt
4. Query LLM via OpenRouter
5. Return formatted answer

### 5. `main.py` - CLI Interface

Command-line tool for testing and development:

```bash
python main.py --query "What is data drift?" --search-method hybrid
```

## 🚀 Deployment

### Streamlit Cloud Deployment

The application is automatically deployed via Streamlit Cloud:

**Deployment Configuration:**
- **Python Version**: 3.12
- **Dependencies**: `requirements.txt`
- **Secrets**: `OPENROUTER_API_KEY` (configured in Streamlit dashboard)
- **Resources**: Standard Streamlit Cloud instance

### Local Development

**1. Install dependencies**
```bash
# From the aihero directory
pip install -r requirements.txt
```

**2. Set up environment variables**
```bash
# Create .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

**3. Run the application**
```bash
streamlit run src/app.py
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter | ✅ Yes | - |
| `LOGS_DIRECTORY` | Directory for logs | ❌ No | `logs/` |

### Search Parameters

Customize search behavior in `search_tools.py`:

```python
# Hybrid search weight
alpha = 0.5  # 0.0 = vector only, 1.0 = text only

# Number of results
num_results = 5

# Chunk parameters
chunk_size = 2000
step = 1000  # Overlap = chunk_size - step
```

## 📊 Performance

### Initialization Time
- **First Load**: ~30-45 seconds
  - Download repo: ~5s
  - Build text index: ~2s
  - Create embeddings: ~25-35s
- **Cached Loads**: <1 second

### Query Response Time
- **Search**: 0.1-0.5 seconds
- **LLM Response**: 2-5 seconds
- **Total**: 2-6 seconds

### Resource Usage
- **Memory**: ~500MB (including embeddings)
- **Storage**: Minimal (cache only)

## 🔍 API Usage

### Programmatic Access

```python
from ingest import ingest_repo, build_text_index
from search_tools import create_embeddings, VectorSearch, hybrid_search
from search_agent import answer_question_manual

# 1. Load data
chunks = ingest_repo("evidentlyai", "docs")
text_index = build_text_index(chunks)

# 2. Build vector search
embeddings = create_embeddings(chunks)
vector_index = VectorSearch()
vector_index.fit(embeddings, chunks)

# 3. Ask questions
answer = answer_question_manual(
    "What is data drift?",
    search_method="hybrid",
    index=text_index,
    vindex=vector_index
)
print(answer)
```

## 🧪 Testing

### Manual Testing

```bash
# Test text search
python main.py --query "monitoring dashboards" --search-method text

# Test vector search
python main.py --query "explain drift detection" --search-method vector

# Test hybrid search (recommended)
python main.py --query "How to evaluate models?" --search-method hybrid
```

### Example Queries

The app includes built-in example questions:
- "What components are required in a test dataset to evaluate AI?"
- "How to run evaluations in Evidently?"
- "How to create monitoring dashboards?"
- "What is data drift detection?"

## 🐛 Troubleshooting

### Common Issues

**1. "OPENROUTER_API_KEY not found"**
```bash
# Solution: Add API key to .env file
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

**2. Slow initial load**
```bash
# Expected: First load takes 30-45s for embedding creation
# Subsequent loads use cached embeddings (<1s)
```

**3. Import errors**
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**4. Out of memory**
```bash
# Solution: Reduce chunk size or use smaller model
# Edit search_tools.py:
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
```

## 🔐 Security

- **API Keys**: Never commit `.env` files
- **Rate Limiting**: OpenRouter handles rate limits
- **Data Privacy**: No user data is stored
- **HTTPS**: Streamlit Cloud provides SSL

## 📈 Monitoring

### Logs

The application can log interactions for debugging:

```python
from logs import log_interaction_to_file

# Logs saved to logs/ directory
log_interaction_to_file(agent, messages)
```

### Metrics to Track

- Query response time
- Search result relevance
- User satisfaction (if feedback added)
- API usage costs

## 🛠️ Extending the Application

### Add New Data Sources

```python
# In app.py or main.py
chunks = ingest_repo("your_org", "your_repo")
```

### Change LLM Provider

```python
# In search_agent.py
openai_client = OpenAI(
    base_url="https://api.openai.com/v1",  # Direct OpenAI
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### Add New Search Methods

```python
# In search_tools.py
def custom_search(query, index, custom_params):
    # Your implementation
    pass
```

## 📚 Related Resources

- **Course**: [AI Agents Email Crash-Course](https://alexeygrigorev.com/aihero/)
- **Articles**: [Medium Collection](https://medium.com/@neidy.tunzine/list/7day-ai-agents-email-crashcourse-a375297638c6)
- **Documentation**: [Evidently AI](https://docs.evidentlyai.com/)
- **OpenRouter**: [API Docs](https://openrouter.ai/docs)

## 🤝 Contributing

Improvements are welcome! Consider:
- Adding a user feedback mechanism
- Implementing conversation memory
- Adding more data sources
- Improving UI/UX
- Adding unit tests

## 📧 Support

- **Issues**: Open a GitHub issue
- **Questions**: Contact via Medium
- **Course Support**: [alexeygrigorev.com/aihero](https://alexeygrigorev.com/aihero/)

---

**Built  as part of the AI Agents Email Crash-Course**

*This application demonstrates production-ready patterns for building intelligent, context-aware AI assistants using modern RAG techniques.*
