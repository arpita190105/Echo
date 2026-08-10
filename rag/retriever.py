"""
Retrieval layer with relevance filtering: given a user query, fetches only
the chunks that are actually similar enough to matter, using cosine distance
plus a keyword-overlap floor as a hard secondary filter (for Latin-script
queries) to catch cases where embedding similarity alone lets a topically
adjacent but wrong chunk through.
"""

import re
import chromadb
from chromadb.utils import embedding_functions
from config import config
from logger import get_logger

logger = get_logger(__name__)

# Devanagari (Hindi) and Bengali unicode ranges — used to detect non-Latin
# script queries, where keyword overlap against English notes will always
# be near-zero even for a correct match, so the hard filter must be skipped.
_NON_LATIN_SCRIPT = re.compile(r"[\u0900-\u097F\u0980-\u09FF]")


def _keyword_overlap_score(query: str, document: str) -> float:
    def tokenize(text):
        return set(re.findall(r"[a-zA-Z\u0900-\u097F\u0980-\u09FF]+", text.lower()))

    query_tokens = tokenize(query)
    doc_tokens = tokenize(document)
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


class Retriever:
    def __init__(self):
        try:
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.rag.embedding_model
            )
            self.client = chromadb.PersistentClient(path=config.rag.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=config.rag.collection_name,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Retriever initialized (collection='{config.rag.collection_name}', "
                f"embedding_model='{config.rag.embedding_model}', space='cosine')"
            )
        except Exception as e:
            logger.exception("Failed to initialize retriever")
            raise RuntimeError(f"Could not initialize RAG retriever: {e}") from e

    def retrieve(self, query: str, subject: str = None, debug: bool = False) -> str:
        try:
            count = self.collection.count()
        except Exception as e:
            logger.error(f"Could not check collection count: {e}")
            return ""

        if count == 0:
            logger.warning("Knowledge base is empty — no notes have been ingested yet.")
            return ""

        where_filter = {"source": subject} if subject else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(config.rag.top_k, count),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Retrieval query failed: {e}")
            return ""

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            logger.info(f"No chunks found at all for query: {query[:60]}")
            return ""

        is_non_latin_query = bool(_NON_LATIN_SCRIPT.search(query))

        candidates = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            keyword_score = _keyword_overlap_score(query, doc)
            candidates.append({"doc": doc, "meta": meta, "distance": dist, "keyword_score": keyword_score})

        candidates.sort(key=lambda c: c["distance"])

        if debug:
            for c in candidates:
                src = c["meta"].get("source", "?") if c["meta"] else "?"
                print(f"  distance={c['distance']:.4f}  keyword_overlap={c['keyword_score']:.2f}  source={src}")

        # Stage 1: cosine distance threshold
        relevant = [c for c in candidates if c["distance"] <= config.rag.distance_threshold]

        # Stage 2: keyword-overlap floor — hard filter, but ONLY for Latin-script
        # queries. Skipped for Hindi/Bengali queries since they won't share
        # tokens with English notes even when the match is correct.
        if not is_non_latin_query:
            relevant = [c for c in relevant if c["keyword_score"] >= config.rag.min_keyword_overlap]

        if not relevant:
            logger.info(
                f"No chunks passed filtering for query '{query[:60]}' "
                f"(threshold={config.rag.distance_threshold}) — falling back to best single match "
                f"(distance={candidates[0]['distance']:.3f})"
            )
            relevant = candidates[:config.rag.min_results]

        relevant = relevant[:config.rag.max_results]

        formatted_chunks = []
        for c in relevant:
            source = c["meta"].get("source", "unknown") if c["meta"] else "unknown"
            formatted_chunks.append(f"[From {source}]\n{c['doc']}")

        logger.info(
            f"Retrieved {len(formatted_chunks)}/{len(documents)} chunks for query: "
            f"{query[:60]} (threshold={config.rag.distance_threshold}, "
            f"non_latin={is_non_latin_query})"
        )
        return "\n\n".join(formatted_chunks)


if __name__ == "__main__":
    import sys
    retriever = Retriever()
    query = sys.argv[1] if len(sys.argv) > 1 else "test query"
    context = retriever.retrieve(query, debug=True)
    print(f"\nRetrieved context:\n{context if context else '(none found)'}")