# PropelAI - Autonomous Proposal Operating System (APOS)

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.11+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-purple.svg" alt="License">
</p>

> **A Stateful Agentic Swarm for Government Proposal Generation**

PropelAI is an AI-powered proposal generation system designed specifically for government contractors. It uses a "Stateful Agentic Swarm" architecture to automate the 80% of proposal work that is administrative "grunt work," allowing proposal managers to focus on strategy.

## 🎯 The Problem We Solve

Small and mid-sized government contractors (SMBs, $2M-$75M revenue) are trapped in a "Valley of Death":
- Too large for set-asides
- Too small to afford the "proposal machines" of large primes (Lockheed, Leidos)
- Proposal managers spend 80% of time on admin tasks, only 20% on strategy

**PropelAI inverts this ratio.**

## 🏗️ Architecture

PropelAI implements the **Orchestrator-Worker pattern** using LangGraph:

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                          │
│              (The Project Manager)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  COMPLIANCE   │ │   STRATEGY    │ │   DRAFTING    │
│    AGENT      │ │    AGENT      │ │    AGENT      │
│ "The Paralegal"│ │"Capture Mgr" │ │ "The Writer"  │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   RED TEAM    │ │   RESEARCH    │ │    HUMAN      │
│    AGENT      │ │    AGENT      │ │   REVIEW      │
│"The Evaluator"│ │"The Librarian"│ │    (HITL)     │
└───────────────┘ └───────────────┘ └───────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/propelai/apos.git
cd apos

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run the Demo

```bash
python demo.py
```

### Start the API Server

```bash
python -m uvicorn api.main:app --reload
```

### Use the CLI

```bash
# Create a new proposal
python cli.py new "DOT Portal Modernization" --client "ACME Corp"

# Upload an RFP
python cli.py upload PROP-ABC12345 ./rfp.pdf

# Run the full workflow
python cli.py shred PROP-ABC12345
python cli.py strategy PROP-ABC12345
python cli.py draft PROP-ABC12345
python cli.py redteam PROP-ABC12345

# Check status
python cli.py status PROP-ABC12345
```

## 📋 The Four "Plays"

### Play 1: Intelligent Shred (RFP Ingestion)
**Goal:** Reduce RFP analysis time from 3 days to <1 hour

The Compliance Agent ("The Paralegal"):
- Parses PDF/DOCX/Excel documents
- Identifies the "Iron Triangle": Requirements (C), Instructions (L), Evaluation (M)
- Extracts all "shall" statements
- Builds the Compliance Matrix
- Maps requirements across sections

### Play 2: Strategy Engine (Win Themes)
**Goal:** Define *why* you win before writing a single word

The Strategy Agent ("The Capture Manager"):
- Analyzes Section M evaluation factors
- Queries past performance for winning patterns
- Develops Win Themes and discriminators
- Performs competitor ghosting analysis
- Creates annotated outline with page allocations

### Play 3: Collaborative Drafting ("The Pen")
**Goal:** Generate compliant, citation-backed narrative text

The Drafting Agent ("The Writer") + Research Agent ("The Librarian"):
- **Zero Hallucination Policy** - Every claim must have a citation
- Queries evidence from past proposals and capabilities
- Generates narrative with hyperlinked citations
- Flags uncited claims as "High Risk" (red underline)

### Play 4: War Room (Red Team Evaluation)
**Goal:** Simulate a government evaluator BEFORE submission

The Red Team Agent ("The Evaluator"):
- Scores using government color ratings (Blue/Green/Yellow/Red)
- Checks compliance against all requirements
- Identifies deficiencies and weaknesses
- Provides specific remediation recommendations
- Maintains the Audit Log (Trust Layer for C-Suite)

## 📁 Project Structure

```
propelai/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── state.py          # ProposalState schema (the heart of the system)
│   ├── orchestrator.py   # LangGraph orchestration
│   └── config.py         # Configuration management
├── agents/
│   ├── __init__.py
│   ├── compliance_agent.py   # RFP shredding
│   ├── strategy_agent.py     # Win theme development
│   ├── drafting_agent.py     # Content generation + Research
│   └── red_team_agent.py     # Proposal evaluation
├── tools/
│   ├── __init__.py
│   └── document_tools.py     # PDF/DOCX parsing
├── api/
│   ├── __init__.py
│   └── main.py               # FastAPI endpoints
├── tests/
│   └── test_agents.py
├── demo.py                   # Full workflow demonstration
├── cli.py                    # Command-line interface
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/proposals` | POST | Create new proposal |
| `/proposals` | GET | List all proposals |
| `/proposals/{id}` | GET | Get proposal details |
| `/proposals/{id}/upload` | POST | Upload RFP document |
| `/proposals/{id}/shred` | POST | Run compliance shredding |
| `/proposals/{id}/strategy` | POST | Generate win strategy |
| `/proposals/{id}/draft` | POST | Generate draft content |
| `/proposals/{id}/redteam` | POST | Run red team evaluation |
| `/proposals/{id}/feedback` | POST | Submit human feedback |
| `/proposals/{id}/export/compliance-matrix` | GET | Export to Excel |
| `/proposals/{id}/audit-log` | GET | Get audit trail |

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 🔐 Security & Compliance

PropelAI is designed for government contractor environments:

- **FedRAMP Ready**: Deployable on Google Assured Workloads
- **Data Sovereignty**: All data stays within the configured VPC
- **Audit Logging**: Complete trace of all AI decisions
- **Human-in-the-Loop**: No proposal submitted without human approval
- **Zero Hallucination**: Every claim must be traceable to source

## 📊 The Data Flywheel

PropelAI gets smarter with every engagement:

1. **Data Capture**: System logs reasoning traces and tool outputs
2. **Human Feedback**: "Brenda's" corrections create high-value training data
3. **Model Improvement**: Agents are fine-tuned on winning patterns
4. **Enhanced Product**: Better first drafts require less editing
5. **User Growth**: Better outcomes attract more clients → more data

This creates a **proprietary moat** that competitors cannot replicate.

## 🛠️ Configuration

Key environment variables:

```bash
# LLM Providers
GOOGLE_API_KEY=xxx          # Primary (Gemini 1.5 Pro - 2M context)
ANTHROPIC_API_KEY=xxx       # Alternative (Claude)
OPENAI_API_KEY=xxx          # Fallback (GPT-4)

# Database
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=xxx

# Vector Store
PINECONE_API_KEY=xxx        # Production
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📈 Roadmap

- [x] **Cycle 1**: Core infrastructure, LangGraph setup, basic shredding
- [x] **Cycle 2**: Drafting with citation enforcement
- [x] **Cycle 3**: Red Team scoring, War Room UI
- [x] **Cycle 4**: Agent-Trace logging, correction signal capture
- [ ] **Cycle 5**: LLM fine-tuning on winning trajectories
- [ ] **Cycle 6**: Multi-tenant SaaS platform
- [ ] **Cycle 7**: Real-time collaboration features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Architecture inspired by [Shipley Process](https://www.shipleyassociates.com/)

---

<p align="center">
  <strong>PropelAI</strong> - Winning Government Contracts with AI
</p>
