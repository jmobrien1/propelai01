# PropelAI - RFP Intelligence Platform

AI-powered RFP analysis and compliance matrix generation for government contractors.

## 🚀 Quick Start

```bash
# Clone and enter directory
cd propelai

# Install dependencies
pip install fastapi uvicorn python-multipart openpyxl python-docx pypdf

# Start the server
./start.sh

# Or manually:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## ✨ Features

### Core Capabilities
- **RFP Parsing**: PDF, DOCX, XLSX support with section detection
- **Requirement Extraction**: Multi-pattern extraction with semantic classification
- **Compliance Matrix**: Color-coded Excel export with priority filtering
- **Amendment Tracking**: Track changes across RFP versions and Q&A responses

### Web Interface
- Drag & drop file upload
- Real-time processing status
- Interactive requirements table with search/filter
- One-click Excel export

## 📁 Project Structure

```
propelai/
├── api/
│   └── main.py              # FastAPI backend
├── web/
│   └── index.html           # React frontend
├── agents/
│   └── enhanced_compliance/ # Core extraction engine
│       ├── agent.py         # Main orchestrator
│       ├── extractor.py     # Requirement extraction
│       ├── parser.py        # Document parsing
│       ├── excel_export.py  # Excel generation
│       └── amendment_processor.py # Change tracking
├── start.sh                 # Startup script
└── requirements.txt
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rfp` | Create new RFP project |
| POST | `/api/rfp/{id}/upload` | Upload documents |
| POST | `/api/rfp/{id}/process` | Start processing |
| GET | `/api/rfp/{id}/status` | Get processing status |
| GET | `/api/rfp/{id}/requirements` | Get requirements |
| GET | `/api/rfp/{id}/export` | Download Excel |
| POST | `/api/rfp/{id}/amendments` | Upload amendment |

## 📊 Tested On

- NIH RFP 75N96025R00004 (228 pages, 629 requirements)
- Illinois IDES UI Claimant Portal RFP (773 requirements, 267 Q&A)
- Federal RFPs with SF30 amendments

## 🛠 Development

```bash
# Run with auto-reload
python -m uvicorn api.main:app --reload

# Run tests
pytest tests/
```

## 📄 License

Proprietary - PropelAI
