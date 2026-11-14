"""Store and search vectors using PostgreSQL + pgvector."""
import json
from typing import Any
import psycopg
from .models import Chunk

PERMITTED_FILTER_KEYS = {"chapter", "topic", "source", "document_id", "page"}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def _vector_str(self, vec: list[float]) -> str:
        if len(vec) != self.dimensions:
            raise ValueError(f"Expected vector of dim {self.dimensions}, got {len(vec)}")
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

    def create_schema(self) -> None:
        """Create table with vector column and source/page/filter metadata."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        chapter TEXT,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_strategy_chapter
                    ON document_chunks (strategy, chapter);
                    """
                )
            conn.commit()

    def count_chunks(self, strategy: str) -> int:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_chunks WHERE strategy = %s;", (strategy,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions and write stable IDs without duplicates.
        Remove obsolete chunks for the ingested (strategy, document_id) pair.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunks count ({len(chunks)}) != vectors count ({len(vectors)})")
        if not chunks:
            return

        doc_ids = list({c.document_id for c in chunks})
        chunk_ids = [c.chunk_id for c in chunks]

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                # Remove obsolete chunks from previous versions of the same document & strategy
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE strategy = %s
                      AND document_id = ANY(%s)
                      AND NOT (chunk_id = ANY(%s));
                    """,
                    (strategy, doc_ids, chunk_ids),
                )

                for chunk, vec in zip(chunks, vectors):
                    vec_lit = self._vector_str(vec)
                    chapter_val = chunk.metadata.get("chapter")
                    cur.execute(
                        """
                        INSERT INTO document_chunks
                            (strategy, chunk_id, document_id, source, page, chapter, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            chapter = EXCLUDED.chapter,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                        """,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chapter_val,
                            chunk.text,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            vec_lit,
                        ),
                    )
            conn.commit()

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k."""
        vec_lit = self._vector_str(query_vector)
        where_clauses = ["strategy = %s"]
        params: list[Any] = [strategy]

        if filters:
            for k, v in filters.items():
                if k not in PERMITTED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {k}")
                if k == "chapter":
                    where_clauses.append("chapter = %s")
                    params.append(str(v))
                elif k == "page":
                    where_clauses.append("page = %s")
                    params.append(int(v))
                elif k in ("source", "document_id"):
                    where_clauses.append(f"{k} = %s")
                    params.append(str(v))
                else:
                    where_clauses.append("metadata ->> %s = %s")
                    params.extend([k, str(v)])

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                chapter,
                text,
                metadata,
                1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """
        query_params = [vec_lit] + params + [vec_lit, top_k]

        results: list[dict[str, Any]] = []
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)
                for row in cur.fetchall():
                    chunk_id, doc_id, source, page, chapter, text, meta, sim = row
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    results.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "source": source,
                            "page": int(page),
                            "chapter": chapter,
                            "text": text,
                            "score": round(float(sim), 4),
                            "metadata": meta,
                        }
                    )
        return results
