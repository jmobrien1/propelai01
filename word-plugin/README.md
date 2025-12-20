# PropelAI Word Plugin v4.0

Microsoft Word Add-in for RFP compliance checking and AI-powered proposal drafting.

## Features

- **Requirement Panel**: Browse and search RFP requirements with source navigation
- **Compliance Checker**: Analyze document against requirements with visual highlighting
- **Draft Assistant**: Generate F-B-P content with AI, insert directly into document
- **Ghosting Library**: Access competitive positioning language
- **Ribbon Commands**: Quick actions without opening the taskpane

## Prerequisites

- Node.js 18+
- Microsoft Word (Desktop or Web)
- PropelAI API running at http://localhost:8000

## Installation

```bash
cd word-plugin
npm install
```

## Development

```bash
# Start development server
npm run dev

# In a separate terminal, sideload the add-in
npm start
```

## Build

```bash
npm run build
```

## Project Structure

```
word-plugin/
├── src/
│   ├── taskpane/           # Main UI components
│   │   ├── components/     # React components
│   │   │   ├── RequirementPanel.tsx
│   │   │   ├── ComplianceChecker.tsx
│   │   │   └── DraftAssistant.tsx
│   │   ├── App.tsx         # Main application
│   │   └── index.tsx       # Entry point
│   ├── commands/           # Ribbon command handlers
│   │   └── commands.ts
│   ├── services/           # API and Word services
│   │   ├── apiClient.ts    # PropelAI API client
│   │   └── wordService.ts  # Word document operations
│   └── types/              # TypeScript definitions
│       └── index.ts
├── assets/                 # Icons and images
├── manifest.xml            # Office Add-in manifest
├── package.json
├── tsconfig.json
└── webpack.config.js
```

## API Endpoints Used

| Endpoint | Description |
|----------|-------------|
| `GET /api/rfp/list` | List all RFPs |
| `GET /api/rfp/{id}` | Get RFP details |
| `GET /api/rfp/{id}/requirements` | Get requirements |
| `GET /api/rfp/{id}/requirements/{req_id}/source` | Trust Gate coordinates |
| `POST /api/rfp/{id}/strategy` | Generate strategy |
| `POST /api/rfp/{id}/draft` | Generate draft |
| `GET /api/rfp/{id}/ghosting-library` | Get ghosting language |

## Ribbon Commands

- **Show Panel**: Opens the PropelAI taskpane
- **Check Compliance**: Analyzes document and highlights matches
- **Insert Draft**: Inserts AI-generated content at cursor
- **Sync RFP**: Synchronizes document with PropelAI

## Configuration

The add-in connects to `http://localhost:8000/api` by default. To change:

```typescript
// In src/services/apiClient.ts
configureApi({ baseUrl: 'https://your-api.com/api' });
```

## License

Proprietary - PropelAI Inc.
