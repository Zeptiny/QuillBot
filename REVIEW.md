# QuillBot — Audit Review: LLM, Prompting & Feature Roadmap

Items identified during the architectural audit that are **not yet implemented**.
Organized by priority.

---

## 1. LLM / Prompting Improvements

### 1.3 New Tools to Expose

| Tool | Source | Purpose |
|---|---|---|
| `check_server_status` | mcsrvstat.us API | Let the LLM check if a user's server is online when diagnosing |
| `get_paper_config_docs` | RAG + PaperMC docs | Dedicated config key lookup |
| `check_plugin_compatibility` | Modrinth version API | Check specific MC version support |
| `get_java_version_info` | Static mapping | MC version → recommended Java version |

### 1.4 Context Window Management

**Current problems:**
- `_truncate_safe(result_text, limit=6000)` per tool × `MAX_TOOL_ROUNDS=4` = up to 24K chars of tool results
- History replays 3 full Q&A pairs verbatim
- `max_tokens=1024` truncates complex answers

**Recommendations:**
```python
# Track cumulative context size
MAX_CONTEXT_CHARS = 20000

# Summarize older history
if history and len(history) > 1:
    summary = f"Resumo: {history[-2]['question']} → {history[-2]['answer'][:200]}"
    messages.append({'role': 'user', 'content': summary})
    # Only keep last exchange verbatim
    messages.append({'role': 'user', 'content': history[-1]['question']})
    messages.append({'role': 'assistant', 'content': history[-1]['answer']})

# Increase max_tokens
max_tokens=2048
```

### 1.5 De-duplicate Tool Results Across Rounds

**Current behavior:**
- If the LLM calls `search_docs` multiple times (in different tool rounds), the same document chunks may appear in both results
- The assistant repeats information redundantly in its final answer

**Action items:**
1. **Verify** — check `docs_rag.py` agentic loop to confirm if duplicate `search_docs` calls occur in practice
2. **If confirmed** — track `all_sources` across rounds and filter duplicates before sending to LLM
3. **Optional enhancement** — add a memo/context field to the LLM stating "avoid repeating these sources you just retrieved"

**Note:** This is lower priority than 1.3/1.4 and depends on runtime observation.

### 1.6 Add Gamemode Knowledge Base

**Rationale:**
- Most common admin questions involve choosing between Survival, Creative, Adventure, and Spectator
- Resource usage, difficulty balancing, and version compatibility vary per gamemode
- Currently the LLM has no structured gamemode reference

**Implementation:**
1. Create a new RAG vector store or static context chunk: `gamemode_reference.md`
   - Survival: description, typical configs, RAM usage, common settings changes
   - Creative: building/testing specific requirements, performance tips
   - Adventure: map design constraints, custom rules
   - Spectator: spectator-only server considerations
2. Include per-gamemode recommendations:
   - Most common MC versions (e.g., Survival = latest; Creative = stable)
   - PaperMC config tuning (e.g., `mob-spawning.per-player`, `difficulty`)
   - Plugin compatibility by gamemode
3. Embed during initial `/reindex` so the LLM can cite it naturally

---

## 2. Feature Roadmap

### 2.1 Interactive Timings/Spark Analysis (High Impact)

Add `/timings` command:
- Accept Spark report URL (`spark.lucko.me`)
- Parse heap summary, tick stats, top plugins from the Spark JSON API
- Feed structured data to the LLM for diagnosis
- Most commonly shared artifact in MC admin channels

### 2.2 Server Configuration Wizard (Medium Impact)

Multi-step `discord.ui.Modal`:
1. Collect: RAM, player count, MC version, server type (Paper/Purpur/Fabric)
2. LLM generates optimized config snippets from RAG context
3. Outputs `server.properties`, `paper-global.yml`, `purpur.yml` changes
4. Converts passive documentation into active, personalized guidance

### 2.3 Plugin Compatibility Checker (Medium Impact)

`/compat <plugin> <version>`:
- Query Modrinth/Hangar version-specific API endpoints
- Cross-reference with known incompatibilities
- Expose as an LLM tool (`check_compatibility`) for natural-language queries
- Addresses the most repetitive question type in support channels


