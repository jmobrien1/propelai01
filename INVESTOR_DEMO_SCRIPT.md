# 🚀 PropelAI - 5-Minute Investor Demo

**Welcome!** This is a quick walkthrough to experience PropelAI's AI-powered proposal intelligence platform.

---

## 🎯 What You'll See

PropelAI helps government contractors:
1. **Extract requirements** from complex RFPs automatically
2. **Chat with RFP documents** like you're talking to an expert
3. **Generate compliance matrices** in seconds (not weeks)
4. **Match capabilities** to opportunities using AI

---

## 📍 Demo Site

**URL:** https://propelai01.onrender.com/

---

## 🎬 Quick Demo (5 Minutes)

### Step 1: Upload Your First RFP Document (1 minute)

1. Go to https://propelai01.onrender.com/
2. You'll see **"Upload RFP Documents"** screen
3. **Option A:** Download sample RFP: https://github.com/GSA/sam-api-examples/raw/master/sample-rfp.pdf
4. **Option B:** Use your own government RFP (PDF, DOCX, Excel)
5. **Drag and drop** the file into the upload area
6. Wait for upload to complete

✅ **Result:** File uploads and you'll see it in the workspace

---

### Step 2: Process & Extract Requirements - The Magic (2 minutes)

1. After upload completes, look in the left sidebar
2. Click **"Compliance Matrix"** or **"Upload RFP"** depending on what you see
3. If there's a **"Process"** button, click it and select **"Best Practices Extraction"**
4. Wait ~30-60 seconds while AI analyzes the document

**What's Happening Behind the Scenes:**
- AI reads entire RFP (500+ pages in minutes)
- Identifies Section L, M, C boundaries
- Extracts "shall/must" requirements
- Categorizes by type (Technical, Management, Compliance)
- Maps to evaluation criteria

✅ **Result:** See extracted requirements with:
- Requirement text
- Section reference (L.4.2.1, C.3.1)
- Priority level
- RFP page numbers

---

### Step 3: Chat with Your RFP - Ask Questions (1 minute)

1. Click the **"Chat"** tab at the top
2. Try these questions:

**Question 1: Basic Info**
```
"What is the contract period of performance?"
```

**Question 2: Requirements**
```
"What are the key deliverables for Phase 1?"
```

**Question 3: Compliance**
```
"What certifications are required for personnel?"
```

**Question 4: Evaluation (Advanced)**
```
"What are the evaluation factors and their weights?"
```

✅ **Result:** AI answers with:
- Direct quotes from RFP
- Source citations (page numbers)
- Context-aware responses

---

### Step 5: Company Library - Match Capabilities (30 seconds)

1. Click **"Library"** in sidebar
2. Click **"Upload"** tab
3. Try uploading:
   - A capability statement
   - Past performance summary
   - Resume/CV
   - Technical whitepaper

4. Go back to **"Chat"** and ask:
```
"Do we have experience with similar intelligence support contracts?"
```

✅ **Result:** AI searches BOTH:
- The uploaded RFP
- Your company library
- Provides match analysis

---

### Step 6: Export Compliance Matrix (Bonus - 30 seconds)

1. Go to **"Artifacts"** tab
2. Click **"Generate Compliance Matrix"**
3. Choose format: **"Best Practices (L-M-C)"**
4. Click **"Download Excel"**

✅ **Result:** Get a production-ready Excel file with:
- Section L Compliance Matrix
- Section M Evaluation Alignment
- Technical Requirements (Section C/PWS)
- Cross-reference mapping

---

## 💡 Key Features You Just Experienced

| Feature | Old Way | PropelAI Way |
|---------|---------|--------------|
| **Requirements Extraction** | 2-3 days manually | 60 seconds with AI |
| **Compliance Matrix** | 1-2 weeks | Download in 30 seconds |
| **RFP Analysis** | Read 500 pages yourself | Ask questions in chat |
| **Capability Matching** | Manual cross-check | AI searches library |
| **Amendment Tracking** | Hope you caught it | Auto-detect conflicts |

---

## 🎯 Use Cases This Demo Covers

### For Capture Managers:
- ✅ Quick bid/no-bid analysis
- ✅ Requirement extraction
- ✅ Gap analysis (what we're missing)

### For Proposal Managers:
- ✅ Compliance matrix generation
- ✅ Section L instruction parsing
- ✅ Page limit extraction

### For Business Development:
- ✅ Company capability matching
- ✅ Past performance search
- ✅ Win theme identification

---

## 🚨 "Wow Factor" Features to Highlight

### 1. **The "Router Brain"** (v4.0)
PropelAI automatically detects RFP type:
- Standard FAR 15? → Apply Section L/M/C logic
- GSA Schedule? → Look for Cover Letter instructions
- OTA/CSO? → Focus on innovation, not compliance
- SBIR/R&D? → Extract scientific evaluation criteria
- Spreadsheet RFP? → Parse Excel questionnaires
- RFI? → Generate capability statement

**No manual configuration needed!**

### 2. **The "War Room" Intelligence**
Ask the AI:
```
"Are there any contradictions between the base RFP and Amendment 2?"
```

AI will:
- Cross-reference all amendments
- Flag conflicts
- Trace requirements to source
- Identify "buried" changes

### 3. **The "Forensic Analysis"**
```
"What are the hidden compliance traps in Section L?"
```

AI scans for:
- "Shall" vs "Should" (mandatory vs preference)
- Buried page limits in narrative text
- Mismatch between L (instructions) and M (evaluation)
- Unreferenced attachments

---

## 📊 What Makes This Different?

**PropelAI vs. Generic AI (ChatGPT/Claude):**

| Generic AI | PropelAI |
|------------|----------|
| Hallucinates facts | Citations with page numbers |
| No document context | RAG: Answers from YOUR RFP |
| One-size-fits-all | 6 specialized modes per RFP type |
| Can't handle 500 pages | Processes full solicitations |
| No structured output | Exports to Excel/Word |

---

## 🎤 Investor Talking Points

### Market Opportunity:
- **$200B+** in US Gov contracts awarded annually
- **15,000+** active solicitations on SAM.gov daily
- Average proposal costs **$50K-500K** in labor
- Win rates: **5-20%** (most companies lose 80%+)

### PropelAI's Value Prop:
- **80% time reduction** in compliance matrix creation
- **$20K-100K savings** per proposal (labor costs)
- **Better compliance** = higher win rates
- **Scale:** One analyst can now handle 10x proposals

### Technical Moat:
- **Specialized AI:** Trained on FAR/DFARS, not generic
- **Router Architecture:** Adapts to 6 different RFP types
- **RAG Integration:** Company library + RFP context
- **Forensic Prompts:** Detects traps human readers miss

---

## 🔒 Enterprise Roadmap (Briefly Mention)

**What You're Seeing:** Version 4.0 (Single-Tenant MVP)

**Coming Soon:**
- Multi-tenant SaaS deployment
- Team collaboration (Capture → Proposal → Review)
- Integrations: Deltek, Microsoft 365, Box/SharePoint
- GovWin opportunity feed integration
- Auto-shred: Upload RFP → Auto-assign sections to writers
- Version control for proposal drafts

---

## ❓ Anticipated Questions & Answers

**Q: "Is this just ChatGPT with a wrapper?"**
A: No. PropelAI uses specialized prompts, multi-mode routing, and RAG architecture. ChatGPT can't:
- Detect RFP type automatically
- Cross-reference amendments
- Generate formatted compliance matrices
- Search company library + RFP simultaneously

**Q: "What about hallucinations?"**
A: Every answer includes source citations (page #, section). Users can verify. RAG prevents hallucinations by only using uploaded documents.

**Q: "Can it handle classified RFPs?"**
A: Current: Commercial cloud (FedRAMP in roadmap). For classified: On-prem deployment option available.

**Q: "What's your defensibility?"**
A: 
1. Specialized prompts (1,000+ lines of FAR logic)
2. Router architecture (6 modes)
3. Network effects (more RFPs → better training)
4. Integration lock-in (Deltek, GovWin)

**Q: "Who's the competition?"**
A: 
- **VisibleThread:** Compliance checking (we do extraction + chat)
- **Loopio:** RFP response automation (we do analysis + requirements)
- **Generic AI:** No specialization for government proposals
- **Current Process:** Manual (80% of market still uses Word/Excel)

---

## 🎁 Demo Tips for Best Results

### Upload Types That Work Great:
✅ Federal RFPs with Section L, M, C
✅ GSA Schedule RFQs
✅ State government solicitations
✅ DoD attachments (J.2, J.3)
✅ Excel questionnaires

### Questions That Show "Wow Factor":
- "What's the page limit for the technical volume?"
- "Summarize the evaluation criteria"
- "What certifications must our staff have?"
- "Are there any amendments that changed requirements?"
- "What's the difference between Section L and Section M?"

### Company Library Demo:
Upload 2-3 diverse docs:
1. Capability statement (shows matching)
2. Past performance (shows relevant experience)
3. Resume (shows personnel alignment)

Then ask:
- "Do we have cloud modernization experience?"
- "Have we worked with this agency before?"
- "Do we have staff with the required certifications?"

---

## 🚀 Call to Action

**For Investors:**
- Schedule a deep-dive demo with our team
- Review our pitch deck and financials
- Connect us to GovCon portfolio companies (natural channel partners)

**For Potential Customers:**
- Sign up for early access (launching paid plans Q2 2025)
- Free pilot for first 10 enterprise customers
- Bring your hardest RFP - we'll analyze it live

---

## 📧 Contact

**Website:** https://propelai01.onrender.com/  
**Demo Request:** [Your contact info]  
**Questions:** Try the chat feature - ask anything about government proposals!

---

## 🎯 One-Sentence Pitch

**"PropelAI is Copilot for government contractors - we use AI to extract requirements, generate compliance matrices, and match capabilities in minutes instead of weeks, saving $20K-100K per proposal."**

---

*Thank you for trying PropelAI! We're building the future of intelligent proposal management.* 🚀
