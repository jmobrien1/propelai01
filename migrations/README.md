# PropelAI Database Migrations

This directory contains Alembic migrations for PropelAI's PostgreSQL database.

## Quick Start

```bash
# Apply all migrations
make db-upgrade

# Create a new migration
make db-migrate

# Rollback last migration
make db-downgrade
```

## Manual Commands

```bash
# View current revision
alembic current

# View migration history
alembic history

# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade <revision>

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision>

# Generate new migration (autogenerate from models)
alembic revision --autogenerate -m "Description of changes"

# Generate empty migration
alembic revision -m "Description of changes"
```

## Environment Variables

The following environment variables can be used to configure the database connection:

- `DATABASE_URL` - Full PostgreSQL connection string (preferred)
- `SQLALCHEMY_DATABASE_URL` - Alternative connection string

Example:
```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/propelai
```

## Migration Guidelines

1. **Always test migrations locally** before applying to production
2. **Never modify existing migrations** that have been applied to production
3. **Include both upgrade and downgrade** functions
4. **Use meaningful names** for migration files
5. **Test rollback** to ensure downgrade works correctly

## pgvector Extension

PropelAI uses pgvector for semantic search. The initial migration enables the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Vector columns use `vector(1536)` to match OpenAI's embedding dimensions.

## Troubleshooting

### Migration fails with "relation already exists"

This usually means the database was created using `init.sql` instead of migrations.
To fix, mark the initial migration as applied:

```bash
alembic stamp 001
```

### "No module named 'api.database'"

Ensure you're running alembic from the project root directory:

```bash
cd /path/to/propelai
alembic upgrade head
```

### Connection refused

Check that PostgreSQL is running and the connection string is correct:

```bash
pg_isready -h localhost -p 5432
```
