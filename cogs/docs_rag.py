import asyncio
import json
import logging
import math
import os
import re

import aiohttp
import discord
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands, tasks
from openai import AsyncOpenAI, RateLimitError

from cogs.plugin_apis import HTTP_HEADERS as _HTTP_HEADERS
from cogs.plugin_apis import search_all as _search_plugins_all
from cogs.utils import PaginatedEmbedView, split_response, truncate_safe as _truncate_safe
from config import (
    CHAT_MODEL,
    COOLDOWN_PER,
    COOLDOWN_RATE,
    DOC_SOURCES,
    DOCS_BASE_URL,
    DOCS_BRANCH,
    EMBEDDING_MODEL,
    GITHUB_API,
    OPENROUTER_API_KEY,
    REINDEX_INTERVAL_HOURS,
    RERANK_MODEL,
    VECTOR_STORE_PATH,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4  # Safety cap on tool-calling iterations

SYSTEM_PROMPT = (
    "<role>\n"
    "Você é o assistente oficial do Miners' Refuge, uma comunidade brasileira de "
    "administradores de servidores Minecraft. Responda sempre em português brasileiro.\n"
    "</role>\n\n"
    "<instructions>\n"
    "1. Use as ferramentas disponíveis para buscar informações antes de responder.\n"
    "2. Se a pergunta for vaga ou ambígua, peça esclarecimentos diretamente — omita chamadas de ferramentas.\n"
    "3. Baseie suas respostas exclusivamente nos dados retornados pelas ferramentas. "
    f"Se nenhuma retornar dados relevantes, diga que não encontrou e sugira visitar {DOCS_BASE_URL}.\n"
    "4. Omita seções de fontes na resposta — as fontes são exibidas automaticamente pela interface.\n"
    "5. Quando útil, termine com uma sugestão de acompanhamento na linha final, prefixada com '💡 '.\n"
    "</instructions>\n\n"
    "<response_format>\n"
    "Seja claro e conciso. Use markdown para formatação. "
    "Forneça passos práticos e acionáveis quando aplicável.\n"
    "</response_format>\n\n"
    "<examples>\n"
    "<example>\n"
    "<user>Como configurar o view-distance para melhorar o desempenho?</user>\n"
    "<assistant>\n"
    "O `view-distance` controla quantos chunks ao redor de cada jogador o servidor processa. "
    "Reduzi-lo é uma das formas mais eficazes de aliviar a carga.\n\n"
    "**`server.properties`**\n"
    "```\nview-distance=6\n```\n\n"
    "**`paper-world-defaults.yml`** (Paper/Purpur)\n"
    "```yaml\nchunks:\n  delay-chunk-unloads-by: 10s\n```\n\n"
    "Para servidores com 20–50 jogadores, valores entre 6 e 8 oferecem bom equilíbrio.\n\n"
    "💡 Quer otimizar também o `simulation-distance`?\n"
    "</assistant>\n"
    "</example>\n"
    "<example>\n"
    "<user>meu server tá com problema</user>\n"
    "<assistant>\n"
    "Para ajudar, preciso de mais detalhes:\n"
    "- Qual versão e tipo de servidor (Paper, Purpur, Fabric)?\n"
    "- O problema é lag, crash ou erro ao conectar?\n"
    "- Você tem um log ou relatório do Spark para compartilhar?\n"
    "</assistant>\n"
    "</example>\n"
    "</examples>"
)

# --- Tool definitions for the LLM (OpenAI function-calling format) ---
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_docs',
            'description': (
                'Pesquisa na documentação do Miners\' Refuge, PaperMC e PurpurMC. '
                'Use para qualquer pergunta sobre configuração, administração ou setup de servidores Minecraft. '
                'Não use para perguntas sobre plugins específicos — use search_plugins para isso.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A consulta de busca em linguagem natural.',
                    },
                    'max_results': {
                        'type': 'integer',
                        'description': (
                            'Número máximo de resultados a retornar (1-12). '
                            'Use valores menores (3-5) para perguntas focadas e maiores (8-12) '
                            'para tópicos amplos que podem abranger múltiplas páginas. Default: 5.'
                        ),
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_plugins',
            'description': (
                'Pesquisa plugins no Modrinth, Hangar e SpigotMC. '
                'Use quando o usuário perguntar sobre plugins, recomendações de plugins ou alternativas. '
                'Não use para perguntas gerais de configuração de servidor — use search_docs para isso.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Nome ou descrição do plugin a pesquisar.',
                    },
                },
                'required': ['query'],
            },
        },
    },
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_frontmatter(content: str) -> dict:
    """Extract key-value pairs from YAML frontmatter (between --- delimiters)."""
    if not content.startswith('---'):
        return {}
    end = content.find('\n---', 3)
    if end == -1:
        return {}
    fm_text = content[3:end]
    result = {}
    for line in fm_text.splitlines():
        m = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if m:
            result[m.group(1)] = m.group(2).strip().strip('"\'')
    return result


def _title_from_path(path: str) -> str:
    """Derive a human-readable title from a file path as a last resort."""
    name = path.rsplit('/', 1)[-1]
    name = re.sub(r'\.(md|mdx)$', '', name)
    name = re.sub(r'[-_]', ' ', name)
    return name.title()


def _compute_doc_url(path: str, base_url: str, url_strip_prefix: str = '') -> str:
    """Convert a repo file path to its docs website URL."""
    base_url = base_url.rstrip('/')
    if url_strip_prefix and path.startswith(url_strip_prefix):
        path = path[len(url_strip_prefix):]
    # Strip extension (suffix only, longest first to avoid .mdx -> x)
    for ext in ('.mdx', '.md'):
        if path.endswith(ext):
            path = path[:-len(ext)]
            break
    # Strip README and index from the final path segment (mdBook / MkDocs conventions)
    for index_name in ('/README', '/index'):
        if path.endswith(index_name):
            path = path[:-len(index_name)]
            break
    if path in ('README', 'index'):
        path = ''
    url = path.rstrip('/')
    return f'{base_url}/{url}' if url else base_url


def path_to_docs_url(path: str) -> str:
    """Convert a MinersRefuge repo file path to its docs URL (backward compat)."""
    return _compute_doc_url(path, DOCS_BASE_URL)


class DocsRAG(commands.Cog):
    """RAG-powered documentation search and Q&A with vector storage and reranking."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client: AsyncOpenAI | None = None
        if OPENROUTER_API_KEY:
            self.client = AsyncOpenAI(
                base_url='https://openrouter.ai/api/v1',
                api_key=OPENROUTER_API_KEY,
            )
        self.session: aiohttp.ClientSession | None = None
        self.chunks: list[dict] = []  # {content, path, title, embedding, source, doc_url}
        self._last_commit_sha: str | None = None
        self._indexing: bool = False
        # TTL cache: max 200 conversations, each expires after 30 min
        self._conversations: TTLCache = TTLCache(maxsize=200, ttl=1800)

    async def cog_load(self):
        self.session = aiohttp.ClientSession(
            headers=_HTTP_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        loaded = self._load_vectors()
        if not loaded:
            await self.index_docs()
        self.periodic_reindex.start()

    async def cog_unload(self):
        self.periodic_reindex.cancel()
        if self.session:
            await self.session.close()

    # --- Cooldown error handler ---

    async def cog_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(
                f'⏳ Aguarde {error.retry_after:.0f}s antes de usar este comando novamente.',
                ephemeral=True,
            )
        else:
            raise error

    # --- Vector Storage ---

    def _save_vectors(self):
        """Persist chunks and embeddings to disk."""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        data = {
            'commit_sha': self._last_commit_sha,
            'chunks': [
                {
                    'content': c['content'],
                    'path': c['path'],
                    'title': c['title'],
                    'embedding': c['embedding'],
                    'source': c.get('source', "Miners' Refuge"),
                    'doc_url': c.get('doc_url', path_to_docs_url(c['path'])),
                }
                for c in self.chunks
            ],
        }
        with open(VECTOR_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Saved %d vectors to %s", len(self.chunks), VECTOR_STORE_PATH)

    def _load_vectors(self) -> bool:
        """Load vectors from disk. Returns True if loaded successfully."""
        if not os.path.exists(VECTOR_STORE_PATH):
            return False
        try:
            with open(VECTOR_STORE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = data.get('chunks', [])
            # Backfill fields absent in older vector store formats
            for chunk in chunks:
                chunk.setdefault('source', "Miners' Refuge")
                chunk.setdefault('doc_url', path_to_docs_url(chunk['path']))
            self.chunks = chunks
            self._last_commit_sha = data.get('commit_sha')
            logger.info(
                "Loaded %d vectors from disk (commit: %s)",
                len(self.chunks),
                self._last_commit_sha,
            )
            return bool(self.chunks)
        except Exception:
            logger.exception("Failed to load vectors from %s", VECTOR_STORE_PATH)
            return False

    # --- Periodic Reindexing ---

    @tasks.loop(hours=REINDEX_INTERVAL_HOURS)
    async def periodic_reindex(self):
        """Check for doc updates and reindex if the primary repo has new commits."""
        try:
            latest_sha = await self._get_latest_commit_sha()
            if latest_sha and latest_sha != self._last_commit_sha:
                logger.info(
                    "New commit detected (%s -> %s), reindexing...",
                    self._last_commit_sha,
                    latest_sha,
                )
                await self.index_docs()
            else:
                logger.info("Docs up to date (commit: %s)", self._last_commit_sha)
        except Exception:
            logger.exception("Error during periodic reindex check")

    @periodic_reindex.before_loop
    async def _wait_for_bot(self):
        await self.bot.wait_until_ready()

    async def _get_latest_commit_sha(self) -> str | None:
        """Fetch the latest commit SHA from the primary docs repo (MinersRefuge/docs)."""
        url = f'{GITHUB_API}/commits/{DOCS_BRANCH}'
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('sha')
        except Exception:
            logger.exception("Failed to fetch latest commit SHA")
        return None

    # --- Indexing ---

    async def _fetch(self, url: str) -> str | None:
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception:
            logger.exception("Failed to fetch %s", url)
        return None

    def _extract_paths(self, summary: str) -> list[str]:
        # Exclude external URLs (http/https) -- only match relative .md paths
        paths = re.findall(r'\((?!https?://)([^)]+\.md)\)', summary)
        return list(dict.fromkeys(paths))  # deduplicate preserving order

    def _extract_title(self, content: str) -> str:
        fm = _parse_frontmatter(content)
        if fm.get('title'):
            return fm['title']
        match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        return match.group(1).strip() if match else ''

    def _chunk_text(self, text: str, path: str, chunk_size: int = 1500) -> list[dict]:
        """Split markdown into chunks by headings, falling back to size-based splits."""
        title = self._extract_title(text)
        sections = re.split(r'(?=^##?\s)', text, flags=re.MULTILINE)
        result = []

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= chunk_size:
                result.append({
                    'content': section,
                    'path': path,
                    'title': title,
                })
            else:
                paragraphs = section.split('\n\n')
                current = []
                current_size = 0
                for para in paragraphs:
                    if current_size + len(para) > chunk_size and current:
                        result.append({
                            'content': '\n\n'.join(current),
                            'path': path,
                            'title': title,
                        })
                        current = []
                        current_size = 0
                    current.append(para)
                    current_size += len(para)
                if current:
                    result.append({
                        'content': '\n\n'.join(current),
                        'path': path,
                        'title': title,
                    })
        return result

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [d.embedding for d in response.data]

    async def _fetch_doc(
        self, path: str, github_raw: str, semaphore: asyncio.Semaphore
    ) -> tuple[str, str | None]:
        """Fetch a single doc file from the given GitHub raw base URL."""
        async with semaphore:
            content = await self._fetch(f'{github_raw}/{path}')
        return path, content

    async def _get_paths_from_tree(
        self, repo: str, branch: str, path_prefix: str = ''
    ) -> list[str]:
        """Use the GitHub tree API to discover .md/.mdx files under an optional prefix."""
        url = f'https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1'
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Tree API returned %d for %s", resp.status, repo)
                    return []
                data = await resp.json()
                tree = data.get('tree', [])
                return [
                    entry['path'] for entry in tree
                    if entry.get('type') == 'blob'
                    and (entry['path'].endswith('.md') or entry['path'].endswith('.mdx'))
                    and (not path_prefix or entry['path'].startswith(path_prefix))
                ]
        except Exception:
            logger.exception("Failed to get file tree for %s", repo)
            return []

    async def _index_source(
        self, source: dict, semaphore: asyncio.Semaphore
    ) -> list[dict]:
        """Index one documentation source and return its text chunks (no embeddings yet)."""
        repo = source['repo']
        branch = source['branch']
        base_url = source['base_url']
        label = source['label']
        github_raw = f'https://raw.githubusercontent.com/{repo}/{branch}'
        url_strip_prefix = source.get('url_strip_prefix', '')
        max_files = source.get('max_files', 200)

        # Discover file paths
        summary_path = source.get('summary')
        if summary_path:
            summary = await self._fetch(f'{github_raw}/{summary_path}')
            if not summary:
                logger.error(
                    "Failed to fetch %s from %s -- skipping source", summary_path, repo
                )
                return []
            paths = self._extract_paths(summary)
            paths.insert(0, 'README.md')
        else:
            path_prefix = source.get('path_prefix', '')
            paths = await self._get_paths_from_tree(repo, branch, path_prefix)

        paths = paths[:max_files]
        if not paths:
            logger.warning("No paths found for source '%s'", label)
            return []

        # Fetch documents concurrently (bounded by the shared semaphore)
        fetch_tasks = [self._fetch_doc(p, github_raw, semaphore) for p in paths]
        results = await asyncio.gather(*fetch_tasks)

        chunks = []
        fetched = 0
        for path, content in results:
            if content:
                fm = _parse_frontmatter(content)
                slug = fm.get('slug')
                if slug:
                    doc_url = f'{base_url}/{slug}'
                else:
                    doc_url = _compute_doc_url(path, base_url, url_strip_prefix)
                for chunk in self._chunk_text(content, path):
                    chunk['source'] = label
                    chunk['doc_url'] = doc_url
                    chunks.append(chunk)
                fetched += 1

        logger.info(
            "Source '%s': fetched %d/%d docs, %d chunks",
            label, fetched, len(paths), len(chunks),
        )
        return chunks

    async def index_docs(self):
        """Fetch all docs from all configured sources and create embeddings."""
        if self._indexing:
            logger.info("index_docs() called while already indexing; skipping")
            return
        self._indexing = True
        try:
            await self._index_docs_inner()
        finally:
            self._indexing = False

    async def _index_docs_inner(self):
        logger.info("Indexing %d documentation source(s)...", len(DOC_SOURCES))

        # All sources share a semaphore to cap total concurrent HTTP fetches
        semaphore = asyncio.Semaphore(5)
        source_tasks = [self._index_source(src, semaphore) for src in DOC_SOURCES]
        source_results = await asyncio.gather(*source_tasks, return_exceptions=True)

        all_chunks = []
        for src, result in zip(DOC_SOURCES, source_results, strict=True):
            if isinstance(result, Exception):
                logger.error("Error indexing source '%s': %s", src['label'], result)
            else:
                all_chunks.extend(result)

        logger.info("Total chunks before embedding: %d", len(all_chunks))
        if not all_chunks:
            return

        # Generate embeddings in batches
        batch_size = 20
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c['content'] for c in batch]
            try:
                embeddings = await self._embed_batch(texts)
                for chunk, emb in zip(batch, embeddings):
                    chunk['embedding'] = emb
            except Exception:
                logger.exception("Failed to embed batch %d", i // batch_size)

        self.chunks = [c for c in all_chunks if 'embedding' in c]
        logger.info("Documentation indexed: %d chunks with embeddings", len(self.chunks))

        # Track primary source commit SHA and persist
        self._last_commit_sha = await self._get_latest_commit_sha()
        self._save_vectors()

    # --- Search with Reranking ---

    async def _rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        """Rerank documents using the reranking model via OpenRouter."""
        try:
            headers = {
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'Content-Type': 'application/json',
            }
            payload = {
                'model': RERANK_MODEL,
                'query': query,
                'documents': [d['content'] for d in documents],
                'top_n': top_n,
            }
            async with self.session.post(
                'https://openrouter.ai/api/v1/rerank',
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get('results', [])
                    results.sort(key=lambda r: r.get('relevance_score', 0), reverse=True)
                    return [documents[r['index']] for r in results[:top_n]]
                else:
                    body = await resp.text()
                    logger.warning("Rerank failed (status %d): %s", resp.status, body)
        except Exception:
            logger.exception("Rerank request failed, falling back to embedding scores")
        # Fallback: return documents as-is (already sorted by cosine similarity)
        return documents[:top_n]

    async def search(self, query: str, top_k: int = 12) -> list[dict]:
        if not self.chunks:
            return []

        try:
            query_emb = (await self._embed_batch([query]))[0]
        except Exception:
            logger.exception("Failed to embed query")
            return []

        scored = []
        for chunk in self.chunks:
            score = cosine_similarity(query_emb, chunk['embedding'])
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top candidates for reranking (wider net than final top_k)
        candidates = [c for _, c in scored[:top_k * 3]]

        # Rerank for better precision
        reranked = await self._rerank(query, candidates, top_n=top_k)
        return reranked

    # --- Slash Commands ---

    @app_commands.command(name='ask', description='Pergunte algo sobre administração de servidores Minecraft')
    @app_commands.checks.cooldown(COOLDOWN_RATE, COOLDOWN_PER)
    @app_commands.describe(
        question='Sua pergunta',
        image='Imagem/screenshot para análise (opcional)',
    )
    async def ask(
        self,
        interaction: discord.Interaction,
        question: str,
        image: discord.Attachment | None = None,
    ):
        if not self.client:
            await interaction.response.send_message(
                '⚠️ Comando indisponível: chave de API não configurada.', ephemeral=True
            )
            return

        if self._indexing:
            await interaction.response.send_message(
                '📚 A documentação está sendo indexada, tente novamente em alguns instantes.',
                ephemeral=True,
            )
            return

        image_url = None
        if image:
            if not image.content_type or not image.content_type.startswith('image/'):
                await interaction.response.send_message(
                    'O arquivo enviado não é uma imagem válida.', ephemeral=True
                )
                return
            image_url = image.url

        await interaction.response.defer(thinking=True)

        try:
            answer, embeds = await self._run_agent(question, image_url=image_url)
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            self._store_conversation(msg.id, question, answer)

        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in /ask command")
            await interaction.followup.send(
                'Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.'
            )

    # --- Tool execution ---

    async def _exec_tool(self, name: str, args: dict) -> tuple[str, list[dict]]:
        """Execute a tool call and return (result_text, source_chunks)."""
        if name == 'search_docs':
            query = args.get('query', '')
            top_k = max(1, min(12, int(args.get('max_results', 5))))
            results = await self.search(query, top_k=top_k)
            if not results:
                return 'Nenhum resultado encontrado na documentação.', []
            parts = []
            for r in results:
                doc_url = r.get('doc_url', path_to_docs_url(r['path']))
                parts.append(f"[Fonte: {r['title']} — {doc_url}]\n{r['content']}")
            return '\n\n---\n\n'.join(parts), results

        if name == 'search_plugins':
            query = args.get('query', '')
            results = await _search_plugins_all(self.session, query)
            if not results:
                return 'Nenhum plugin encontrado.', []
            lines = []
            for r in results[:6]:
                versions = ', '.join(str(v) for v in r.get('versions', [])[:3]) or '—'
                lines.append(
                    f"**{r['name']}** ({r['source']}) — {r.get('description', '')}\n"
                    f"Downloads: {r.get('downloads', 0):,} | Versões: {versions}\n"
                    f"URL: {r['url']}"
                )
            return '\n\n'.join(lines), []

        return f'Ferramenta desconhecida: {name}', []

    # --- Agentic loop ---

    async def _run_agent(
        self,
        question: str,
        history: list[dict] | None = None,
        image_url: str | None = None,
    ) -> tuple[str, discord.Embed]:
        """Run the LLM with tool-calling in a loop until it produces a final answer."""
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

        if history:
            for h in history[-3:]:
                messages.append({'role': 'user', 'content': h['question']})
                messages.append({'role': 'assistant', 'content': h['answer']})

        if image_url:
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            })
        else:
            messages.append({'role': 'user', 'content': question})

        all_sources: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                max_tokens=1024,
            )

            choice = response.choices[0]

            # No tool calls → final answer
            if not choice.message.tool_calls:
                break

            # Append the assistant message with tool calls
            messages.append(choice.message)

            # Execute each tool call
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                result_text, sources = await self._exec_tool(tc.function.name, args)
                all_sources.extend(sources)

                messages.append({
                    'role': 'tool',
                    'tool_call_id': tc.id,
                    'content': _truncate_safe(result_text, limit=6000),
                })
        else:
            # If we exhausted rounds, get a final response without tools
            response = await self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            choice = response.choices[0]

        answer = choice.message.content or 'Não foi possível gerar uma resposta.'

        # Build source links (attached to first page only)
        sources_value: str | None = None
        if all_sources:
            seen_paths: set[str] = set()
            source_lines: list[str] = []
            for r in all_sources:
                if r['path'] not in seen_paths:
                    seen_paths.add(r['path'])
                    doc_url = r.get('doc_url', path_to_docs_url(r['path']))
                    title = r['title'] or _title_from_path(r['path'])
                    source_label = r.get('source', "Miners' Refuge")
                    source_lines.append(f'• [{title}]({doc_url}) — {source_label}')
                if len(source_lines) >= 8:
                    break
            if source_lines:
                sources_value = '\n'.join(source_lines)

        pages = split_response(answer)
        total = len(pages)
        footer_base = (
            f"Documentação • {DOCS_BASE_URL} • "
            "💬 Responda a esta mensagem para continuar a conversa"
        )
        embeds: list[discord.Embed] = []
        for i, page_text in enumerate(pages):
            e = discord.Embed(
                title=f'❓ {question}' if i == 0 else '',
                description=page_text,
                color=discord.Color.blue(),
            )
            if i == 0 and sources_value:
                e.add_field(
                    name='📄 Fontes da Documentação',
                    value=sources_value,
                    inline=False,
                )
            e.set_footer(
                text=f"Página {i + 1}/{total} • {footer_base}" if total > 1 else footer_base
            )
            embeds.append(e)

        return answer, embeds

    def _store_conversation(self, message_id: int, question: str, answer: str):
        """Store a conversation exchange for follow-up replies."""
        self._conversations[message_id] = {
            'question': question,
            'answer': answer,
            'history': [],
        }

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle reply-based follow-up conversations."""
        if message.author.bot:
            return
        if not message.reference or not message.reference.message_id:
            return

        ref_id = message.reference.message_id
        conv = self._conversations.get(ref_id)
        if not conv:
            return

        follow_up_question = message.content.strip()
        if not follow_up_question and not message.attachments:
            return

        image_url = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith('image/'):
                image_url = att.url
                break

        if not follow_up_question:
            follow_up_question = 'Analise esta imagem.'

        async with message.channel.typing():
            try:
                history = conv.get('history', []).copy()
                history.append({'question': conv['question'], 'answer': conv['answer']})

                answer, embeds = await self._run_agent(
                    follow_up_question, history=history, image_url=image_url
                )
                if len(embeds) == 1:
                    reply = await message.reply(embed=embeds[0])
                else:
                    reply = await message.reply(
                        embed=embeds[0], view=PaginatedEmbedView(embeds)
                    )

                self._conversations[reply.id] = {
                    'question': follow_up_question,
                    'answer': answer,
                    'history': history,
                }
            except RateLimitError:
                await message.reply(
                    '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
                )
            except Exception:
                logger.exception("Error in follow-up reply")
                await message.reply(
                    'Ocorreu um erro ao processar sua pergunta. Tente novamente.'
                )

    @app_commands.command(name='reindex', description='Re-indexar a documentação (Admin)')
    @app_commands.checks.has_permissions(administrator=True)
    async def reindex(self, interaction: discord.Interaction):
        if self._indexing:
            await interaction.response.send_message(
                '📚 Já há uma indexação em andamento, aguarde a conclusão.',
                ephemeral=True,
            )
            return
        await interaction.response.defer(thinking=True)
        await self.index_docs()
        await interaction.followup.send(
            f'✅ Documentação re-indexada! ({len(self.chunks)} chunks)'
        )


async def setup(bot: commands.Bot):
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set -- DocsRAG cog will not be loaded")
        return
    await bot.add_cog(DocsRAG(bot))
