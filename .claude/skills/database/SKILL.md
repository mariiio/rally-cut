---
name: database
description: Manage PostgreSQL database and Prisma ORM - migrations, schema changes, data exploration. Use for database schema updates or data inspection. (project)
allowed-tools: Bash, Read, Edit
---

# RallyCut Database Management

## Prisma Commands (run from api/)

```bash
npx prisma migrate dev      # Create and apply migration
npx prisma migrate deploy   # Apply pending migrations (prod)
npx prisma generate         # Regenerate Prisma client
npx prisma studio           # Visual database browser
npx prisma db push          # Push schema without migration (dev only)
```

## Schema Location

`api/prisma/schema.prisma`

## Key Models

- `User` → tier (FREE/PREMIUM), quotas
- `Session` → editing session container
- `Video` → uploaded video metadata
- `Rally` → detected rally with timestamps
- `Highlight` → user-created highlight reel
- `ExportJob` → video export tracking

## Docker Commands

```bash
cd api && docker-compose up -d    # Start PostgreSQL
cd api && docker-compose down     # Stop
cd api && docker-compose down -v  # Stop + delete data
```

## Common Tasks

### Add new field
1. Edit `api/prisma/schema.prisma`
2. Run `npx prisma migrate dev --name add_field_name`
3. Prisma client auto-regenerates

### Reset database
```bash
npx prisma migrate reset  # Drops all data, re-applies migrations
```
