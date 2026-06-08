# wiki_corpus

Place retrieval files here for RAG methods (`naive_rag`, `drag`, `mad`, etc.):

- `wiki18_100w.jsonl` (document corpus)
- `e5_flat_inner.index` (FAISS index)

Expected paths used by default config:

- `./wiki_corpus/wiki18_100w.jsonl`
- `./wiki_corpus/e5_flat_inner.index`

If you use different filenames/locations, pass overrides in `main.py`:

```powershell
python .\main.py --index_path "D:\path\to\your.index" --corpus_path "D:\path\to\your.jsonl"
```
