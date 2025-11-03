# PPTGen-AI

AI-powered service to generate PowerPoint (PPT) and PDF documents from web content.  
FastAPI backend + Streamlit frontend. Uses SearxNG for meta-search and Crawl4AI for crawling, then a RAG pipeline to produce content. Output files use basic formatting (no custom themes).

## Flow Diagram
<img width="529" height="242" alt="image" src="https://github.com/user-attachments/assets/97adf039-b559-4c8e-872d-ec0310dbf821" />

Some steps or components may not have been included yet and are potential areas for future improvement.

## Key features
- Web search (SearxNG) + crawl (Crawl4AI) → extract article content
- Retrieval-Augmented Generation (RAG) to create summaries and slide content
- Generate downloadable plain PDF and PPT files
- REST API: /search, /ask, /generate, /download/{filename}
- Streamlit frontend for user interaction
- API key protection and structured logging
- Docker Compose setup for local development

## RAG (Retrieval Augmented Generation) Components
### Embeddings & Vectors
- **Embeddings Model**: Sentence Transformers from HuggingFace
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Dimension: 384
  - Framework: PyTorch

### Vector Storage & Similarity Search
- **Vector Database**: FAISS (Facebook AI Similarity Search)
  - Index Type: IndexFlatL2
  - Distance Metric: L2 (Euclidean)
  - In-memory storage

### Content Processing
- **Tokenizer**: HuggingFace Transformers
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Max Length: 512 tokens
- **Text Chunking**: Basic paragraph splitting with overlap

### Generation
- **LLM**: Groq API
- **LLM Model**: llama-3.3-70b-versatile
- **Prompt Templates**: Custom templates in `lib/prompt.py`
- **Output Format**: Structured JSON for PDF/PPT generation
  
## Quick Setup
1. **Configure SearxNG**:
   ```bash
   # Copy settings template
   cp searxng/settings.yml.example searxng/settings.yml

   # Generate random secret key and update settings.yml
   python -c "import secrets; print(secrets.token_hex(32))" > secretkey.txt
   sed -i "s/ultrasecretkey/$(cat secretkey.txt)/g" searxng/settings.yml
   rm secretkey.txt
   ```

2. **Set up environment**:
   ```bash
   # Copy env template
   cp .env.example .env

   # Generate random API key
   python -c "import secrets; print(f'API_KEY={secrets.token_urlsafe(32)}')" >> .env

   # Set Groq API key (replace YOUR_KEY with actual key)
   echo "GROQ_API_KEY=YOUR_KEY" >> .env
   ```

3. **Start services**:
   ```bash
   docker-compose up --build
   ```

For Windows PowerShell:
```powershell
# Copy config files
Copy-Item searxng/settings.yml.example searxng/settings.yml
Copy-Item .env.example .env

# Generate and set secret key
$secretKey = -join ((48..57) + (97..122) | Get-Random -Count 32 | % {[char]$_})
(Get-Content searxng/settings.yml) -replace 'ultrasecretkey', $secretKey | Set-Content searxng/settings.yml

# Generate and append API key
$apiKey = -join ((48..57) + (97..122) | Get-Random -Count 32 | % {[char]$_})
Add-Content .env "`nAPI_KEY=$apiKey"

# Add Groq API key (replace YOUR_KEY)
Add-Content .env "`nGROQ_API_KEY=YOUR_KEY"
```

## Services & Endpoints
- API: http://127.0.0.1:6789
- SearxNG: http://127.0.0.1:8080
- Streamlit frontend: http://127.0.0.1:8501

## Example API usage
Search (note trailing slash):
```
curl -X POST "http://127.0.0.1:8080/search/" -d "q=ai&format=json" -H "Content-Type: application/x-www-form-urlencoded"
```

Generate (API):
```
curl -X POST "http://127.0.0.1:6789/generate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: <YOUR_KEY>" \
  -d '{"query":"AI in education","topic":"AI for teachers","type":"ppt","limit":5}'
```

Download:
```
curl -O -J -H "x-api-key: <YOUR_KEY>" "http://127.0.0.1:6789/download/<filename>"
```

## Project layout
- app/ — FastAPI application (crawler, rag, file generator, main)
- frontend/ — Streamlit frontend
- searxng/ — SearxNG config files (settings.yml templates, uwsgi.ini)
- shared/ — config and logger
- output/ — generated files (mounted into containers)

## Third-Party Components
- **SearxNG**: Privacy-respecting meta search engine
  - Source: https://github.com/searxng/searxng
  - License: AGPL-3.0
- **Crawl4AI**: Web crawling library
  - Source: https://github.com/unclecode/crawl4ai
  - License: Apache-2.0 license
- **SearCrawl**: SearXNG and Crawl4AI configuration
  - Source: https://github.com/Bclound/searCrawl
  - License: MIT License
- **Groq**: Large Language Model API
  - Docs: https://console.groq.com/docs
  - Terms: https://groq.com/legal

## Contact
Report issues via repository issue tracker. Include logs and request samples for faster diagnosis.
- Email (for personal things): salsyahirah@gmail.com
- Visit my linkedin [here](https://www.linkedin.com/in/salsabilasyahirah).
- Read how I build this here.
