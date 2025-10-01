# AI Agents Crash Course 🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/)
[![Medium Articles](https://img.shields.io/badge/Medium-Articles-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@neidy.tunzine/list/7day-ai-agents-email-crashcourse-a375297638c6)

A comprehensive 7-day journey to build a production-ready AI agent from scratch. This repository is part of [Alex Grigorev's AI Agents Email Crash-Course](https://alexeygrigorev.com/aihero/), designed to take you from basic data ingestion to a fully deployed intelligent assistant.

## 📖 About This Project

This project demonstrates how to build an AI assistant that understands and answers questions about the [Evidently AI](https://github.com/evidentlyai/evidently) documentation. Through progressive development over 7 days, you'll learn:

- **Data Ingestion**: Download and parse GitHub repositories
- **Text Chunking**: Split documents into searchable chunks
- **Search Systems**: Implement text, vector, and hybrid search
- **AI Integration**: Connect to LLMs via OpenRouter
- **Evaluation**: Measure and optimize search quality
- **Deployment**: Deploy a working Streamlit application

## 🎯 Learning Objectives

This course is designed to help you:
- ✅ Build a portfolio-ready AI project
- ✅ Master data ingestion from real-world sources
- ✅ Implement multiple search strategies (text, semantic, hybrid)
- ✅ Integrate with modern LLM APIs
- ✅ Evaluate and optimize AI system performance
- ✅ Deploy production applications
- ✅ Earn a certificate of completion

## 🚀 Live Demo

Try the deployed application: [FAQ Agent - Evidently AI Docs](https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/)

## 📚 Course Structure

### Daily Progress

Each day builds upon the previous one:

| Day | Topic | Key Concepts | Notebook |
|-----|-------|--------------|----------|
| **Day 1** | Data Ingestion | GitHub API, Frontmatter parsing, Markdown processing | [`course_day1.ipynb`](aihero/course/course_day1.ipynb) |
| **Day 2** | Text Chunking | Sliding windows, Document splitting, Overlap strategies | [`course_day2.ipynb`](aihero/course/course_day2.ipynb) |
| **Day 3** | Text Search | BM25, TF-IDF, Keyword matching | [`day3_text_search.ipynb`](aihero/course/day3_text_search.ipynb) |
| **Day 4** | Vector & Hybrid Search | Embeddings, Semantic search, Result fusion | [`day3_text_vector_hybrid_search.ipynb`](aihero/course/day3_text_vector_hybrid_search.ipynb) |
| **Day 5** | LLM Integration | Agents, Tool calling, RAG patterns | [`day4_agents.ipynb`](aihero/course/day4_agents.ipynb) |
| **Day 6** | Evaluation | Hit rate, MRR, Precision@k metrics | [`day5_evaluation.ipynb`](aihero/course/day5_evaluation.ipynb) |
| **Day 7** | Deployment | Streamlit, Production patterns, Monitoring | See [`aihero/src/`](aihero/src/) |

### 📝 Detailed Articles

Read the full breakdown of each day on Medium:
- [7-Day AI Agents Crash Course Collection](https://medium.com/@neidy.tunzine/list/7day-ai-agents-email-crashcourse-a375297638c6)

## 🛠️ Technology Stack

- **Language**: Python 3.12+
- **Framework**: Streamlit
- **Search**: minsearch (text), sentence-transformers (vectors)
- **LLM**: DeepSeek via OpenRouter
- **Data**: GitHub repositories (Evidently AI docs)
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- UV package manager (or pip)
- OpenRouter API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-agents-crash-course.git
   cd ai-agents-crash-course
   ```

2. **Install dependencies**
   ```bash
   # Using UV (recommended)
   cd aihero
   uv sync
   
   # Or using pip
   pip install -r aihero/requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cd aihero
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

4. **Run the application**
   ```bash
   streamlit run aihero/src/app.py
   ```

## 🎓 Course Exercises

### Jupyter Notebooks

Explore the day-by-day learning path in the [`aihero/course/`](aihero/course/) directory:

```bash
cd aihero/course
jupyter notebook
```

Each notebook is self-contained and demonstrates the concepts for that day.

### Command-Line Interface

Test the search and answer functionality:

```bash
cd aihero
python src/main.py --query "What is data drift?" --search-method hybrid
```

## 🏗️ Project Structure

```
ai-agents-crash-course/
├── aihero/
│   ├── course/              # Daily Jupyter notebooks
│   │   ├── course_day1.ipynb
│   │   ├── course_day2.ipynb
│   │   ├── day3_text_search.ipynb
│   │   ├── day3_text_vector_hybrid_search.ipynb
│   │   ├── day4_agents.ipynb
│   │   └── day5_evaluation.ipynb
│   ├── src/                 # Production code
│   │   ├── app.py          # Streamlit application
│   │   ├── ingest.py       # Data ingestion
│   │   ├── search_tools.py # Search implementations
│   │   ├── search_agent.py # LLM integration
│   │   └── main.py         # CLI interface
│   ├── pyproject.toml
│   └── requirements.txt
├── README.md
└── pyproject.toml
```

## 🔍 Key Features

### 1. **Multi-Strategy Search**
   - **Text Search**: Fast keyword-based retrieval
   - **Vector Search**: Semantic similarity matching
   - **Hybrid Search**: Best of both worlds with tunable weights

### 2. **Intelligent Answer Generation**
   - RAG (Retrieval-Augmented Generation) pattern
   - Context-aware responses
   - Source attribution

### 3. **Interactive Web Interface**
   - Real-time chat interface
   - Search method selection
   - Example questions

### 4. **Evaluation Framework**
   - Hit rate calculation
   - Mean Reciprocal Rank (MRR)
   - Precision@k metrics
   - Hyperparameter optimization

## 📊 Performance Metrics

Based on evaluation with 10 test queries on Evidently AI documentation:

| Search Method | Hit Rate | MRR | Precision@5 |
|--------------|----------|-----|-------------|
| Text Only | 60% | 0.550 | 40% |
| Vector Only | 100% | 0.695 | 48% |
| **Hybrid** | **90%** | **0.670** | **50%** |

## 🤝 Contributing

This is a learning project, but contributions are welcome! Feel free to:

- Report bugs
- Suggest improvements
- Share your own implementations
- Add new features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alex Grigorev** for the [AI Agents Email Crash-Course](https://alexeygrigorev.com/aihero/)
- **Evidently AI** for their excellent documentation used as the knowledge base
- **OpenRouter** for providing easy access to various LLMs
- **Streamlit** for the deployment platform

## 📧 Contact

- **Author**: Neidy Tunzine
- **Medium**: [@neidy.tunzine](https://medium.com/@neidy.tunzine)
- **Course Articles**: [7-Day AI Agents Crash Course](https://medium.com/@neidy.tunzine/list/7day-ai-agents-email-crashcourse-a375297638c6)

## 🎓 Certificate

Upon completion of the course, you'll receive a certificate validating your AI development skills and demonstrating your ability to:
- Build end-to-end AI applications
- Implement advanced search systems
- Integrate with modern LLM APIs
- Deploy production-ready applications

---

**Ready to start building?** Check out the [course notebooks](aihero/course/) or jump straight to the [deployed application](https://neidynm-ai-agents-crash-course-aiherosrcapp-wxdzh1.streamlit.app/)!
