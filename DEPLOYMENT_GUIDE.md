# PropelAI Deployment Guide

## Data Storage Configuration

### File-Based JSON Database

PropelAI uses a simple, stable file-based JSON storage system located in:

**Primary Storage**: `outputs/data/` (relative to project root)
**Fallback Storage**: `/tmp/propelai_data/` (if primary fails)

### Storage Files

- `rfps.json` - RFP metadata, requirements, and state
- `chat_history.json` - Chat message logs by RFP ID  
- `library.json` - Company library document metadata

### Path Resolution Logic

The database automatically handles path resolution:

1. **Development/Local**: Uses `{project_root}/outputs/data/`
2. **Production/Render**: Attempts `{project_root}/outputs/data/`
3. **Fallback**: Uses `/tmp/propelai_data/` if write permission fails

### Render.com Deployment

#### Option 1: Persistent Disk (Recommended)

```yaml
# render.yaml
services:
  - type: web
    name: propelai
    env: docker
    disk:
      name: propelai-data
      mountPath: /app/outputs
      sizeGB: 1
```

This ensures data persists across deployments.

#### Option 2: Environment Variable

Set `DATA_DIR` environment variable to specify custom path:

```bash
DATA_DIR=/opt/render/project/outputs/data
```

#### Option 3: Fallback Mode (Temporary)

If no persistent disk is configured, the app will use `/tmp/propelai_data/` which is always writable but **NOT persistent** across container restarts.

### Backup & Restore

**Backup:**
```bash
cp outputs/data/*.json backup/
```

**Restore:**
```bash
cp backup/*.json outputs/data/
```

### Permissions

The app requires write access to:
- `outputs/data/` directory (or fallback to `/tmp`)
- Individual JSON files for atomic writes

If running in restricted environment, the fallback to `/tmp` ensures the app always starts successfully.

### Monitoring

Check logs for storage location:
```
[DB] Using primary storage: /app/outputs/data
```

Or fallback warning:
```
[DB] WARNING: Could not write to /app/outputs/data
[DB] Using fallback storage: /tmp/propelai_data
[DB] Data will NOT persist across container restarts!
```

### Migration from MongoDB

This version replaces MongoDB with file-based storage. All data is now stored in simple JSON files that can be:
- Versioned in Git (if desired)
- Backed up with simple file copy
- Inspected with any text editor
- Migrated with standard file operations

No database server required!
