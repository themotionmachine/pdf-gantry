# Deep Research: Optimal Chunk-Level Embedding Strategy for Academic PDF Retrieval with Nomic Embed V2 + sqlite-vec

## Context

I'm building a single-user CLI tool (Python 3.11, macOS/Apple Silicon) that manages ~2,000 academic PDFs. The stack:
- **Text extraction**: PyMuPDF4LLM (produces markdown with paragraph breaks, headers, etc.)
- **Embedding model**: Nomic Embed V2 MoE (`nomic-ai/nomic-embed-text-v2-moe`), 768 dimensions, 8K token context, asymmetric search (uses `search_document:` / `search_query:` prefixes)
- **Vector store**: sqlite-vec (vec0 virtual tables in SQLite), brute-force KNN
- **Current approach**: One embedding per document (title + abstract + first ~8K tokens of text). This truncates long papers and makes findings/methodology in the back half invisible to semantic search.

I want to move to chunk-level embeddings. The corpus is ~1,700 digital PDFs averaging 15-25 pages each. Estimated chunk count: 10,000-25,000 vectors depending on chunk size.

## Research Questions

### 1. Optimal chunk size and boundary strategy for Nomic Embed V2

Nomic Embed V2 supports 8,192 tokens of context, but what passage length produces the highest-quality embeddings for *retrieval* (not generation)?

Specifically:
- What chunk sizes (in tokens or characters) were used in Nomic's training data / MTEB benchmarks? Is there a sweet spot where the model's retrieval performance peaks?
- How does retrieval quality (measured by NDCG@10 or similar) degrade as chunk size shrinks below or grows beyond that sweet spot?
- For academic text specifically (dense, argumentative, citation-heavy prose), does sentence-level, paragraph-level, or fixed-window chunking produce better retrieval? Are there published comparisons?
- What is the empirical consensus on overlap size? Is 10-15% overlap standard, or do some strategies (e.g. sentence-boundary-aware chunking) eliminate the need for overlap entirely?

I'm particularly interested in evidence from MTEB, BEIR, or LoTTE benchmarks, and any Nomic-specific guidance from their documentation or papers.

### 2. PyMuPDF4LLM markdown structure as a chunking primitive

PyMuPDF4LLM converts PDFs to markdown. I want to know whether its paragraph/section boundaries are reliable enough to serve as chunk boundaries for academic papers.

- What delimiters does PyMuPDF4LLM use between paragraphs, sections, and pages in its markdown output? (e.g., double newlines, `#` headers, `---` page breaks)
- How does it handle two-column layouts, footnotes, figure captions, tables, and equations — do these produce clean paragraph breaks or fragmented text?
- Are there known failure modes where PyMuPDF4LLM merges or splits paragraphs incorrectly for academic PDFs?
- Would it be better to chunk the raw markdown by structural markers (headers, double newlines) or to use a secondary sentence/paragraph splitter on the extracted text?

### 3. sqlite-vec performance characteristics at 10k-25k vectors

sqlite-vec uses brute-force (flat) KNN by default. At my scale:

- What are measured query latencies for KNN search over 10k, 25k, 50k, and 100k 768-dimensional float32 vectors in sqlite-vec?
- Is there a vector count threshold where query latency becomes noticeable (>100ms) for 768-d vectors?
- Does sqlite-vec support any approximate nearest neighbor indexing (IVF, HNSW, quantization), or is it brute-force only? If brute-force only, at what scale does it become a bottleneck?
- Are there any known issues with sqlite-vec and WAL mode, concurrent reads, or large INSERT batches?
- What is the disk space cost per vector (768 × 4 bytes = 3KB, plus overhead)?

I want to know if I can naively scale to 25k vectors without architecture changes, or if I need to plan for approximate search.

### 4. Chunk-to-document aggregation for search results

When search operates at chunk level but results should be presented at document level, how should chunk scores be aggregated?

- **Max-pooling** (best chunk score represents the document): When does this work well, and when does it produce false positives from spurious chunk matches?
- **Mean-pooling** (average all chunk scores): Does this dilute strong matches in long documents?
- **Top-k pooling** (average of top 3 chunk scores): Is there evidence this outperforms max or mean?
- **Reciprocal Rank Fusion across chunks**: Does RRF make sense for within-document aggregation, or is it only appropriate for cross-system fusion?
- **ColBERT-style late interaction / MaxSim**: Is this feasible with sqlite-vec, or does it require a different retrieval architecture?

What do production RAG systems (LlamaIndex, LangChain, Pinecone documentation, Weaviate documentation) actually recommend and implement? What does the academic IR literature (e.g., SIGIR, ECIR papers from 2023-2026) say about passage-to-document aggregation?

### 5. Hybrid architectures: document-level + chunk-level embeddings

Some systems maintain both a document-level summary embedding and chunk-level detail embeddings, using the document embedding for coarse ranking and chunk embeddings for fine-grained retrieval.

- Is there published evidence that a two-tier approach (doc summary + chunks) outperforms pure chunk-level retrieval for corpora of 1,000-5,000 documents?
- What about "parent document retrieval" (retrieve chunk, return parent document) — does this pattern have measurable advantages over returning chunks directly?
- For a single-user local tool where latency is not a primary concern, is the complexity of a two-tier system justified, or does pure chunk-level with good aggregation suffice?
- If I keep the existing doc-level embedding alongside new chunk embeddings, how should I combine the two signal types at query time?

## Constraints

- Single user, local macOS, M1 Max with 64GB RAM
- sqlite-vec is the only vector store (no Postgres, no cloud services)
- Nomic Embed V2 MoE is the embedding model (not switching models)
- Python 3.11, sentence-transformers for model loading
- Embedding time is not a concern — correctness and retrieval quality matter more than speed
- The tool is invoked by AI agents (Claude Code) as much as by humans — output token efficiency matters

## Desired Output Format

For each research question, provide:
1. **Evidence summary** with citations (papers, benchmarks, documentation)
2. **Concrete recommendation** for my specific context (corpus size, model, vector store)
3. **Implementation parameters** (exact chunk sizes, overlap values, aggregation formulas) I can code against directly

End with a unified recommendation that synthesizes across all five questions into a single coherent chunking + retrieval strategy.
