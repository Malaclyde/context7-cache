# @malaclyde/context7-cache-oc

An [OpenCode](https://opencode.ai) plugin that caches [Context7](https://context7.com) documentation lookups using SQLite, vector embeddings, and full-text search (FTS5).

## Installation

```bash
npm install @malaclyde/context7-cache-oc
```

## Configuration

Add the plugin to your `opencode.json`:

```json
{
  "plugin": [
    ["@malaclyde/context7-cache-oc", {
      "API_KEY": "ctx7sk-...",
      "DB_LOCATION": "global"
    }]
  ]
}
```

### Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `API_KEY` | `string` | omitted | Context7 API key. Omit for free tier access. |
| `DB_LOCATION` | `"global" \| "local"` | `"global"` | Cache storage location: `global` → `~/.cache/opencode/context7-cache/`, `local` → `<project>/.opencode/context7-cache/` |

## How it works

The plugin provides two tools — `resolve_library_id` and `query_docs` — that wrap Context7's API. Results are cached to avoid redundant network calls:

- **Search results** (library ID lookups) are stored in a SQLite table with a 30-day TTL.
- **Docs responses** are saved as `.md` files; their queries are embedded and stored alongside an FTS5 index. Subsequent queries with similar intent (cosine similarity ≥ 0.50) return the cached result instead of hitting the API.

## License

MIT
