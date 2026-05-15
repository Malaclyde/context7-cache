import path from "path"
import os from "os"
import { mkdirSync } from "fs"
import { tool } from "@opencode-ai/plugin"
import type { Plugin } from "@opencode-ai/plugin"
import { Database } from "bun:sqlite"
import { pipeline } from "@xenova/transformers"

const API_BASE = "https://context7.com/api"
const CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000
const COSINE_THRESHOLD = 0.50

const STOP_WORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
  "may", "might", "can", "shall", "to", "of", "in", "for", "on", "with",
  "at", "by", "from", "as", "into", "through", "during", "before", "after",
  "above", "below", "between", "out", "off", "over", "under", "again",
  "further", "then", "once", "here", "there", "when", "where", "why",
  "how", "all", "each", "every", "both", "few", "more", "most", "other",
  "some", "such", "no", "nor", "not", "only", "own", "same", "so",
  "than", "too", "very", "just", "because", "but", "and", "or", "if",
  "while", "about", "up", "down", "what", "which", "who", "whom", "this",
  "that", "these", "those", "am", "it", "its", "i", "me", "my", "myself",
  "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
  "he", "him", "his", "himself", "she", "her", "hers", "herself",
  "they", "them", "their", "theirs", "themselves",
])

export default (async function context7Plugin(ctx, options) {
  const apiKey = options?.API_KEY as string | undefined
  const dbLocation = (options?.DB_LOCATION as string) ?? "global"

  const CACHE_DIR = dbLocation === "local"
    ? path.join(ctx.directory, ".opencode", "context7-cache")
    : path.join(os.homedir(), ".cache", "opencode", "context7-cache")

  function ensureDir(dir: string): void {
    try { mkdirSync(dir, { recursive: true }) } catch {}
  }

  function getDb(): Database {
    ensureDir(CACHE_DIR)
    const db = new Database(`${CACHE_DIR}/cache.db`, { create: true })
    db.run("PRAGMA journal_mode = WAL")

    db.run(`CREATE TABLE IF NOT EXISTS search_cache (
      query TEXT NOT NULL,
      library_name TEXT NOT NULL,
      result_json TEXT NOT NULL,
      fetched_at TEXT NOT NULL,
      PRIMARY KEY (query, library_name)
    )`)

    const cols = db.query("PRAGMA table_info(docs_cache)").all() as { name: string }[]
    if (!cols.some(c => c.name === "query_embedding")) {
      db.run("DROP TABLE IF EXISTS docs_cache")
      db.run("DROP TABLE IF EXISTS docs_fts")
    }

    db.run(`CREATE TABLE IF NOT EXISTS docs_cache (
      library_id TEXT NOT NULL,
      query TEXT NOT NULL,
      query_embedding BLOB,
      file_path TEXT NOT NULL,
      fetched_at TEXT NOT NULL,
      PRIMARY KEY (library_id, query)
    )`)

    db.run(`CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
      query,
      content='docs_cache',
      content_rowid='rowid'
    )`)

    db.run(`CREATE TRIGGER IF NOT EXISTS docs_fts_ai AFTER INSERT ON docs_cache BEGIN
      INSERT INTO docs_fts(rowid, query) VALUES (new.rowid, new.query);
    END`)
    db.run(`CREATE TRIGGER IF NOT EXISTS docs_fts_ad AFTER DELETE ON docs_cache BEGIN
      INSERT INTO docs_fts(docs_fts, rowid, query) VALUES('delete', old.rowid, old.query);
    END`)
    db.run(`CREATE TRIGGER IF NOT EXISTS docs_fts_au AFTER UPDATE ON docs_cache BEGIN
      INSERT INTO docs_fts(docs_fts, rowid, query) VALUES('delete', old.rowid, old.query);
      INSERT INTO docs_fts(rowid, query) VALUES (new.rowid, new.query);
    END`)

    return db
  }

  function isExpired(fetchedAt: string): boolean {
    return Date.now() - new Date(fetchedAt).getTime() > CACHE_TTL_MS
  }

  function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0, na = 0, nb = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
  }

  function cacheFilename(libraryId: string, query: string): string {
    let hash = 5381
    const input = `${libraryId}\x00${query}`
    for (let i = 0; i < input.length; i++) {
      hash = ((hash << 5) + hash) + input.charCodeAt(i)
      hash = hash & hash
    }
    return Math.abs(hash).toString(36)
  }

  function fts5Query(query: string): string | null {
    const terms = query
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(t => t.length > 1 && !STOP_WORDS.has(t))

    if (terms.length === 0) return null
    return terms.map(t => `"${t}"`).join(" OR ")
  }

  function formatSearchResults(json: string): string {
    try {
      const data = JSON.parse(json)
      if (!data.results || data.results.length === 0) {
        return data.error || "No libraries found matching the provided name."
      }
      const lines: string[] = ["Available Libraries:\n"]
      for (const r of data.results) {
        lines.push(`Library: ${r.title}`)
        lines.push(`  ID: ${r.id}`)
        lines.push(`  Description: ${r.description || "N/A"}`)
        lines.push(`  Code Snippets: ${r.totalSnippets ?? "N/A"}`)
        lines.push(`  Source Reputation: ${r.trustScore != null ? (r.trustScore >= 7 ? "High" : r.trustScore >= 4 ? "Medium" : "Low") : "Unknown"}`)
        lines.push(`  Benchmark Score: ${r.benchmarkScore ?? "N/A"}`)
        if (r.versions && r.versions.length > 0) {
          lines.push(`  Versions: ${r.versions.join(", ")}`)
        }
        lines.push("")
      }
      return lines.join("\n")
    } catch {
      return json
    }
  }

  async function apiGet(path: string): Promise<string | null> {
    const headers: Record<string, string> = {}
    if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`
    try {
      const res = await fetch(`${API_BASE}${path}`, { headers })
      if (!res.ok) return null
      return await res.text()
    } catch {
      return null
    }
  }

  let extractor: any = null

  async function getEmbedding(text: string): Promise<Float32Array> {
    if (!extractor) {
      extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
    }
    const result = await extractor(text, { pooling: "mean", normalize: true })
    return result.data as Float32Array
  }

  return {
    tool: {
      resolve_library_id: tool({
        description: "Resolves package/product name to Context7 library ID. Returns matching libraries with IDs, descriptions, snippet counts, trust scores. Call before query_docs unless you already have library ID. For library/framework questions, use context7 first: returns official, version-specific documentation with correct API patterns.",
        args: {
          query: tool.schema.string().describe("The question or task you need help with. Used to rank results by relevance."),
          libraryName: tool.schema.string().describe("Library name to search for. Use proper punctuation (e.g. 'Next.js' not 'nextjs')."),
        },
        async execute(args) {
          const db = getDb()

          const cached = db.query(
            "SELECT result_json, fetched_at FROM search_cache WHERE query = ? AND library_name = ?"
          ).get(args.query, args.libraryName) as { result_json: string; fetched_at: string } | null

          if (cached && !isExpired(cached.fetched_at)) {
            return formatSearchResults(cached.result_json)
          }

          const responseText = await apiGet(
            `/v2/libs/search?query=${encodeURIComponent(args.query)}&libraryName=${encodeURIComponent(args.libraryName)}`
          )

          if (!responseText) {
            if (cached) return formatSearchResults(cached.result_json)
            return "No libraries found matching the provided name."
          }

          try {
            db.run(
              "INSERT OR REPLACE INTO search_cache (query, library_name, result_json, fetched_at) VALUES (?, ?, ?, ?)",
              [args.query, args.libraryName, responseText, new Date().toISOString()]
            )
          } catch {}

          return formatSearchResults(responseText)
        },
      }),

      query_docs: tool({
        description: "Retrieves up-to-date documentation and code examples from Context7. Returns authoritative docs with correct API patterns. Enable researchMode on retry if first answer insufficient. 30-day semantic cache — near-instant on cache hit.",
        args: {
          libraryId: tool.schema.string().describe("Exact Context7-compatible library ID (e.g., '/vercel/next.js', '/vercel/next.js/v14.3.0-canary.87'). Obtain via resolve_library_id or from the user."),
          query: tool.schema.string().describe("The question or task to get documentation for. Be specific (e.g. 'How to set up authentication with JWT in Express.js')."),
          researchMode: tool.schema.boolean().optional().describe("Enable deep research: spins up agents that read source repos and run web search, then synthesizes a fresh answer. Use on retry if the first answer wasn't sufficient."),
        },
        async execute(args) {
          const db = getDb()
          const filename = cacheFilename(args.libraryId, args.query)
          const filePath = `${CACHE_DIR}/${filename}.md`

          const queryEmbedding = await getEmbedding(args.query)
          const embeddingBuf = Buffer.from(queryEmbedding.buffer)

          const ftsTerms = fts5Query(args.query)
          if (ftsTerms) {
            const candidates = db.query(`
              SELECT DISTINCT d.file_path, d.fetched_at, d.query_embedding
              FROM docs_fts
              JOIN docs_cache d ON docs_fts.rowid = d.rowid
              WHERE d.library_id = ? AND docs_fts MATCH ?
              ORDER BY rank
            `).all(args.libraryId, ftsTerms) as { file_path: string; fetched_at: string; query_embedding: Buffer | null }[]

            let best: { file_path: string; similarity: number } | null = null
            for (const c of candidates) {
              if (isExpired(c.fetched_at)) continue
              if (!c.query_embedding) continue
              const cachedEmbedding = new Float32Array(
                c.query_embedding.buffer,
                c.query_embedding.byteOffset,
                c.query_embedding.byteLength / Float32Array.BYTES_PER_ELEMENT
              )
              const sim = cosineSimilarity(queryEmbedding, cachedEmbedding)
              if (sim >= COSINE_THRESHOLD && (!best || sim > best.similarity)) {
                best = { file_path: c.file_path, similarity: sim }
              }
            }
            if (best) {
              try { return await Bun.file(best.file_path).text() } catch {}
            }
          }

          let path = `/v2/context?query=${encodeURIComponent(args.query)}&libraryId=${encodeURIComponent(args.libraryId)}&type=txt`
          if (args.researchMode) path += "&researchMode=true"

          const docsText = await apiGet(path)
          if (!docsText) {
            const fallback = db.query(
              "SELECT file_path FROM docs_cache WHERE library_id = ? ORDER BY fetched_at DESC"
            ).get(args.libraryId) as { file_path: string } | null
            if (fallback) {
              try { return await Bun.file(fallback.file_path).text() } catch {}
            }
            return "Documentation not found for this library. Verify the library ID is correct."
          }

          try {
            await Bun.write(filePath, docsText)
            db.run(
              "INSERT OR REPLACE INTO docs_cache (library_id, query, query_embedding, file_path, fetched_at) VALUES (?, ?, ?, ?, ?)",
              [args.libraryId, args.query, embeddingBuf, filePath, new Date().toISOString()]
            )
          } catch {}

          return docsText
        },
      }),
    },
  }
}) satisfies Plugin
