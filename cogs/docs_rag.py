import asyncio
import hashlib
import json
import logging
import os
import re

import aiohttp
import discord
import numpy as np
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands, tasks
from openai import AsyncOpenAI, RateLimitError

from cogs.plugin_apis import HTTP_HEADERS as _HTTP_HEADERS
from cogs.plugin_apis import search_all as _search_plugins_all
from cogs.spark_parser import (
    AVAILABLE_SECTIONS as _SPARK_SECTIONS,
    SparkReport,
    build_detail as _spark_build_detail,
    build_summary as _spark_build_summary,
)
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
    SPARK_MODEL,
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
    "3. Baseie suas respostas nos dados retornados pelas ferramentas. "
    f"Se nenhuma retornar dados relevantes, diga que não encontrou e sugira visitar {DOCS_BASE_URL}.\n"
    "4. Omita seções de fontes na resposta — as fontes são exibidas automaticamente pela interface.\n"
    "5. Quando útil, termine com uma sugestão de acompanhamento na linha final, prefixada com '💡 '.\n"
    "</instructions>\n\n"
    "<response_format>\n"
    "Seja claro e conciso. A resposta será exibida no Discord — siga estas regras de formatação:\n"
    "- **Negrito** com **texto**, _itálico_ com _texto_, `código inline` com backticks.\n"
    "- Blocos de código com triple backtick e linguagem: ```yaml, ```properties, ```json.\n"
    "- Listas com - ou 1. 2. 3.\n"
    "- Títulos de seção com **Negrito** ou __Sublinhado__ — NÃO use # headings (não renderizam).\n"
    "- NUNCA use tabelas markdown (pipes |) — o Discord não as renderiza. "
    "Substitua tabelas por listas com negrito: **Chave**: valor.\n"
    "- Para comparações lado a lado, use blocos de código ou listas separadas por seção.\n"
    "</response_format>\n\n"
    "<examples>\n"
    "<example>\n"
    "<user>Como configurar o view-distance para melhorar o desempenho?</user>\n"
    "<assistant>\n"
    "O `view-distance` controla quantos chunks ao redor de cada jogador o servidor processa. "
    "Reduzi-lo é uma das formas mais eficazes de aliviar a carga.\n\n"
    "**`server.properties`**\n"
    "```properties\nview-distance=6\n```\n\n"
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
# Appended to SYSTEM_PROMPT when a Spark report is active in the session.
# Following Anthropic best-practice: XML tags separate role, instructions and
# tool guidance; numbered steps encode the required diagnostic reasoning order.
SPARK_SYSTEM_PROMPT_SUFFIX = (
    "\n\n"
    "<spark_analysis_context>\n"
    "<role_extension>\n"
    "Você também é um especialista em diagnóstico de desempenho de servidores "
    "Minecraft. Um relatório do Spark Profiler foi carregado para esta conversa. "
    "Seu objetivo é identificar a causa raiz de problemas de desempenho com base "
    "nos dados do relatório. Seja preciso e fundamentado nos dados — não invente "
    "valores de configuração.\n"
    "</role_extension>\n\n"
    "<diagnostic_protocol>\n"
    "Ao analisar um relatório Spark, siga esta ordem:\n\n"
    "1. **TPS** — O servidor está laggando? (< 20 TPS confirma lag; ≈ 20 TPS com "
    "MSPT max alto indica spikes intermitentes.)\n"
    "2. **MSPT median vs max** — Lag constante (median > 50 ms) ou spikes "
    "intermitentes (median ok, max >> 50 ms)?\n"
    "3. **Perfil de lag spike** — Se `tick_length_threshold_ms > 0`, os hotspots "
    "mostram APENAS os ticks laggados, NÃO a carga normal. Adapte o diagnóstico: "
    "100 % num perfil de spike ≠ servidor sempre a 100 %.\n"
    "4. **`waitForNextTick()`** — Qual % do thread principal é sono?\n"
    "   • ≥ 20 % → capacidade de sobra\n"
    "   • < 20 % → trabalhando muito, vulnerável a spikes\n"
    "   • < 5 % → provavelmente laggando constantemente\n"
    "   • ≥ 80 % → servidor ocioso — perfil pode ter sido coletado no momento errado\n"
    "5. **GC** — Pausas de GC coincidem com picos de MSPT? "
    "Chame `get_spark_detail(\"jvm\")` se necessário.\n"
    "6. **Hotspots** — Identifique o método com maior `self_pct`. "
    "Chame `get_spark_detail(\"hotspots\")` para a árvore completa.\n"
    "7. **Configuração** — Use `get_config_key(file, key)` para verificar o valor "
    "atual da configuração suspeita.\n"
    "8. **Recomendação fundamentada** — Chame `search_docs(query, source=\"PaperMC\")` "
    "para obter valores recomendados e justificativa da documentação oficial. "
    "Nunca invente valores de configuração.\n"
    "</diagnostic_protocol>\n\n"
    "<tool_guidance>\n"
    "Use as ferramentas proativamente e na ordem indicada acima. "
    "Prefira `get_config_key` quando precisar de apenas uma chave. "
    "Use `get_spark_detail` para dados completos de uma seção. "
    "Sempre baseie recomendações de configuração em `search_docs`, "
    "não em conhecimento de treinamento.\n"
    "</tool_guidance>\n\n"
    "<spark_response_format>\n"
    "Ao apresentar a análise do relatório Spark, use este padrão de formatação Discord:\n"
    "- Liste métricas chave com **Negrito**: valor (ex: **TPS**: 18.5, **MSPT mediana**: 62 ms).\n"
    "- NUNCA use tabelas markdown (pipes |) — o Discord não as renderiza.\n"
    "- Para múltiplos hotspots ou configurações, use listas com `-` e `código inline`.\n"
    "- Agrupe por seção usando **Negrito** como título (ex: **Diagnóstico**, **Recomendações**).\n"
    "- Valores de configuração sempre em bloco de código com a linguagem correta.\n"
    "</spark_response_format>\n"
    "</spark_analysis_context>"
)
# --- Tool definitions for the LLM (OpenAI function-calling format) ---
_SOURCE_LABELS = [src['label'] for src in DOC_SOURCES]
_SOURCE_LABELS_STR = ', '.join(f'"{s}"' for s in _SOURCE_LABELS)

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_docs',
            'description': (
                'Pesquisa na documentação indexada. '
                f'Fontes disponíveis: {_SOURCE_LABELS_STR}. '
                'Use para qualquer pergunta sobre configuração, administração ou setup de servidores Minecraft. '
                'Use o parâmetro `source` para restringir a busca a uma fonte específica quando a pergunta '
                'claramente pertence a um projeto concreto (ex: Spark para profiling, PaperMC para configurações '
                'Paper). Omita `source` para perguntas gerais ou que cruzam múltiplas documentações. '
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
                    'source': {
                        'type': 'string',
                        'enum': _SOURCE_LABELS,
                        'description': (
                            f'Filtrar por fonte de documentação específica ({_SOURCE_LABELS_STR}). '
                            'Use apenas quando a pergunta é claramente específica de um projeto. '
                            'Se omitido, pesquisa em todas as fontes.'
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

# Additional tools injected only when a Spark report is active in the session.
_SPARK_SECTIONS_STR = ', '.join(f'"{s}"' for s in _SPARK_SECTIONS)
SPARK_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_spark_detail',
            'description': (
                'Retorna dados detalhados de uma seção específica do relatório Spark '
                'atualmente carregado. Use para obter informações além do resumo inicial. '
                f'Seções disponíveis: {_SPARK_SECTIONS_STR}. '
                'Use "hotspots" para identificar gargalos na árvore de chamadas completa. '
                'Use "jvm" para verificar GC e flags da JVM. '
                'Use "profiler" para estatísticas TPS/MSPT por janela de tempo. '
                'Use "configs:<arquivo>" para obter um arquivo de configuração específico '
                '(ex: "configs:server.properties", "configs:paper/").'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'section': {
                        'type': 'string',
                        'description': (
                            'Nome da seção a retornar. Use "configs:<arquivo>" para '
                            'configurações por arquivo (ex: "configs:spigot.yml").'
                        ),
                    },
                },
                'required': ['section'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_config_key',
            'description': (
                'Lê uma única chave de configuração do servidor sem buscar o arquivo inteiro. '
                'Use quando só precisar verificar um valor específico antes de recomendar uma mudança. '
                'Mais eficiente que get_spark_detail para consultas pontuais.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'file': {
                        'type': 'string',
                        'description': (
                            'Nome do arquivo de configuração '
                            '(ex: "server.properties", "spigot.yml", "paper/").'
                        ),
                    },
                    'key': {
                        'type': 'string',
                        'description': 'Nome da chave de configuração a consultar.',
                    },
                },
                'required': ['file', 'key'],
            },
        },
    },
]


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
        self._emb_matrix: np.ndarray | None = None  # (N, dim) float32 for vectorized search
        # TTL cache: max 200 conversations, each expires after 30 min
        self._conversations: TTLCache = TTLCache(maxsize=200, ttl=1800)
        # Per-user follow-up cooldown (same period as slash commands)
        self._followup_cd: TTLCache = TTLCache(maxsize=500, ttl=COOLDOWN_PER)

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

    def _rebuild_matrix(self) -> None:
        """Rebuild the numpy embedding matrix from current chunks."""
        if not self.chunks:
            self._emb_matrix = None
            return
        self._emb_matrix = np.array(
            [c['embedding'] for c in self.chunks], dtype=np.float32
        )

    def _save_vectors(self):
        """Persist chunk metadata to JSON and embeddings to a numpy binary file."""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        meta = {
            'commit_sha': self._last_commit_sha,
            'chunks': [
                {
                    'content': c['content'],
                    'path': c['path'],
                    'title': c['title'],
                    'source': c.get('source', "Miners' Refuge"),
                    'doc_url': c.get('doc_url', path_to_docs_url(c['path'])),
                    # embeddings stored separately in .npy
                }
                for c in self.chunks
            ],
        }
        with open(VECTOR_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
        npy_path = os.path.splitext(VECTOR_STORE_PATH)[0] + '.npy'
        embeddings = np.array([c['embedding'] for c in self.chunks], dtype=np.float32)
        np.save(npy_path, embeddings)
        self._emb_matrix = embeddings
        logger.info("Saved %d vectors to %s + %s", len(self.chunks), VECTOR_STORE_PATH, npy_path)

    def _load_vectors(self) -> bool:
        """Load chunk metadata from JSON and embeddings from numpy binary file."""
        if not os.path.exists(VECTOR_STORE_PATH):
            return False
        npy_path = os.path.splitext(VECTOR_STORE_PATH)[0] + '.npy'
        try:
            with open(VECTOR_STORE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunks = data.get('chunks', [])

            if os.path.exists(npy_path):
                # New binary format
                embeddings = np.load(npy_path)
                if len(embeddings) != len(chunks):
                    logger.warning(
                        "Embedding count mismatch (%d vs %d chunks), will reindex",
                        len(embeddings), len(chunks),
                    )
                    return False
                for chunk, emb in zip(chunks, embeddings):
                    chunk['embedding'] = emb
            elif chunks and 'embedding' in chunks[0]:
                # Old JSON-with-embeddings format — migrate on load
                logger.info("Migrating vector store from JSON to binary format...")
                embeddings = np.array([c.pop('embedding') for c in chunks], dtype=np.float32)
                for chunk, emb in zip(chunks, embeddings):
                    chunk['embedding'] = emb
            else:
                logger.warning("No embeddings found in vector store, will reindex")
                return False

            for chunk in chunks:
                chunk.setdefault('source', "Miners' Refuge")
                chunk.setdefault('doc_url', path_to_docs_url(chunk['path']))
            self.chunks = chunks
            self._last_commit_sha = data.get('commit_sha')
            self._rebuild_matrix()
            logger.info(
                "Loaded %d vectors from disk (commit: %s)",
                len(self.chunks),
                self._last_commit_sha,
            )
            # If loaded from old format, immediately save in new binary format
            if not os.path.exists(npy_path):
                self._save_vectors()
            return bool(self.chunks)
        except Exception:
            logger.exception("Failed to load vectors from %s", VECTOR_STORE_PATH)
            return False

    # --- Periodic Reindexing ---

    @tasks.loop(hours=REINDEX_INTERVAL_HOURS)
    async def periodic_reindex(self):
        """Check for doc updates across all sources and reindex if anything changed."""
        try:
            latest_sha = await self._get_composite_sha()
            if latest_sha and latest_sha != self._last_commit_sha:
                logger.info(
                    "Doc source changes detected (%s -> %s), reindexing...",
                    self._last_commit_sha,
                    latest_sha,
                )
                await self.index_docs()
            else:
                logger.info("All doc sources up to date (composite: %s)", self._last_commit_sha)
        except Exception:
            logger.exception("Error during periodic reindex check")

    @periodic_reindex.before_loop
    async def _wait_for_bot(self):
        await self.bot.wait_until_ready()

    async def _get_composite_sha(self) -> str | None:
        """Fetch latest commit SHAs from all configured doc sources and return a composite hash."""
        shas = []
        for source in DOC_SOURCES:
            url = f'https://api.github.com/repos/{source["repo"]}/commits/{source["branch"]}'
            try:
                async with self.session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        sha = data.get('sha', '')
                        if sha:
                            shas.append(f'{source["repo"]}:{sha}')
                    else:
                        logger.warning(
                            "Could not fetch commit SHA for %s (status %d)",
                            source['repo'], resp.status,
                        )
            except Exception:
                logger.exception("Failed to fetch commit SHA for %s", source['repo'])
        if not shas:
            return None
        return hashlib.md5(':'.join(sorted(shas)).encode()).hexdigest()

    async def _get_latest_commit_sha(self) -> str | None:
        """Fetch the latest commit SHA from the primary docs repo (MinersRefuge/docs).

        Kept for backward compatibility; prefer _get_composite_sha for reindex checks.
        """
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
        self._rebuild_matrix()

        # Track composite commit SHA across all sources and persist
        self._last_commit_sha = await self._get_composite_sha()
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

    async def search(self, query: str, top_k: int = 12, source_filter: str | None = None) -> list[dict]:
        if not self.chunks or self._emb_matrix is None:
            return []

        # Apply optional source filter
        if source_filter:
            indices = [i for i, c in enumerate(self.chunks) if c.get('source') == source_filter]
            if not indices:
                return []
            idx_arr = np.array(indices)
            chunks = [self.chunks[i] for i in indices]
            emb_matrix = self._emb_matrix[idx_arr]
        else:
            chunks = self.chunks
            emb_matrix = self._emb_matrix

        try:
            query_emb = (await self._embed_batch([query]))[0]
        except Exception:
            logger.exception("Failed to embed query")
            return []

        query_arr = np.array(query_emb, dtype=np.float32)
        dots = emb_matrix @ query_arr
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_arr)
        with np.errstate(invalid='ignore', divide='ignore'):
            scores = np.where(norms > 0, dots / norms, 0.0)

        top_indices = np.argsort(scores)[::-1][:top_k * 3]
        candidates = [chunks[i] for i in top_indices]

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

    async def _exec_tool(
        self,
        name: str,
        args: dict,
        spark_report: SparkReport | None = None,
    ) -> tuple[str, list[dict]]:
        """Execute a tool call and return (result_text, source_chunks)."""
        if name == 'search_docs':
            query = args.get('query', '')
            top_k = max(1, min(12, int(args.get('max_results', 5))))
            source_filter = args.get('source') or None
            results = await self.search(query, top_k=top_k, source_filter=source_filter)
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

        if name == 'get_spark_detail':
            if spark_report is None:
                return 'Nenhum relatório Spark carregado para esta conversa.', []
            section = args.get('section', '')
            return _spark_build_detail(spark_report, section), []

        if name == 'get_config_key':
            if spark_report is None:
                return 'Nenhum relatório Spark carregado para esta conversa.', []
            file_name = args.get('file', '')
            key = args.get('key', '')
            cfg = spark_report.configs.get(file_name)
            if cfg is None:
                available = ', '.join(spark_report.configs.keys()) or 'none'
                return (
                    f'Arquivo "{file_name}" não encontrado. '
                    f'Disponíveis: {available}'
                ), []
            if isinstance(cfg, dict):
                if key not in cfg:
                    return f'Chave "{key}" não encontrada em "{file_name}".', []
                value = cfg[key]
                return f'{file_name}/{key} = {json.dumps(value)}', []
            return f'{file_name} = {cfg}', []

        return f'Ferramenta desconhecida: {name}', []

    # --- Agentic loop ---

    async def _run_agent(
        self,
        question: str,
        history: list[dict] | None = None,
        image_url: str | None = None,
        spark_report: SparkReport | None = None,
        title: str | None = None,
    ) -> tuple[str, list[discord.Embed]]:
        """Run the LLM with tool-calling in a loop until it produces a final answer.

        When ``spark_report`` is provided the agent uses ``SPARK_MODEL``,
        receives the report summary as a synthetic prior exchange, and gains
        access to the ``get_spark_detail`` / ``get_config_key`` Spark tools.
        """
        # Build system prompt ----------------------------------------------------
        system_content = SYSTEM_PROMPT
        if spark_report is not None:
            system_content += SPARK_SYSTEM_PROMPT_SUFFIX
            if spark_report.tick_length_threshold_ms > 0:
                system_content += (
                    '\n\n<lag_spike_warning>\n'
                    f'ATENÇÃO: Este é um PERFIL DE LAG SPIKE '
                    f'(--only-ticks-over {spark_report.tick_length_threshold_ms}ms). '
                    'Os hotspots representam a CAUSA dos spikes, NÃO a carga normal do servidor. '
                    '100% num perfil de lag spike ≠ o servidor está sempre a 100% — significa que '
                    'aquele método estava presente em todos os ticks que laggaram. '
                    'Não sugira otimizações gerais de desempenho como resposta primária.\n'
                    '</lag_spike_warning>'
                )

        messages: list[dict] = [
            {
                'role': 'system',
                # Content-array format enables per-message cache_control breakpoints
                # supported by both Gemini (SPARK_MODEL) and Anthropic providers via
                # OpenRouter. The breakpoint is placed after the full system prompt so
                # the entire static instruction block is cached (5-min TTL by default).
                'content': [
                    {
                        'type': 'text',
                        'text': system_content,
                        'cache_control': {'type': 'ephemeral'},
                    }
                ],
            }
        ]

        # Inject Spark report summary as a synthetic prior exchange --------------
        # Per Anthropic long-context guidance: put data before the user question.
        if spark_report is not None:
            summary_text = (
                '[Relatório Spark carregado]\n\n'
                + _spark_build_summary(spark_report)
            )
            messages.append({
                'role': 'user',
                # Cache the report summary so the agentic tool-call loop reuses
                # it from cache on every subsequent round without re-billing.
                'content': [
                    {
                        'type': 'text',
                        'text': summary_text,
                        'cache_control': {'type': 'ephemeral'},
                    }
                ],
            })
            messages.append({
                'role': 'assistant',
                'content': (
                    'Entendido. Analisei o resumo do relatório Spark e estou pronto '
                    'para diagnosticar.'
                ),
            })

        # Replay conversation history --------------------------------------------
        if history:
            for h in history[-3:]:
                messages.append({'role': 'user', 'content': h['question']})
                messages.append({'role': 'assistant', 'content': h['answer']})

        # Current user message ---------------------------------------------------
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

        # Choose model and tool set based on session type ------------------------
        model = SPARK_MODEL if spark_report is not None else CHAT_MODEL
        active_tools = TOOLS + SPARK_TOOLS if spark_report is not None else TOOLS

        all_sources: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=active_tools,
                max_tokens=2048,
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

                result_text, sources = await self._exec_tool(
                    tc.function.name, args, spark_report=spark_report
                )
                all_sources.extend(sources)

                messages.append({
                    'role': 'tool',
                    'tool_call_id': tc.id,
                    'content': _truncate_safe(result_text, limit=6000),
                })
        else:
            # If we exhausted rounds, get a final response without tools
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
            )
            choice = response.choices[0]

        answer = choice.message.content or 'Não foi possível gerar uma resposta.'

        # Build source links (attached to first page only) -----------------------
        sources_value: str | None = None
        if all_sources:
            seen_paths: set[str] = set()
            source_lines: list[str] = []
            for r in all_sources:
                if r['path'] not in seen_paths:
                    seen_paths.add(r['path'])
                    doc_url = r.get('doc_url', path_to_docs_url(r['path']))
                    doc_title = r['title'] or _title_from_path(r['path'])
                    source_label = r.get('source', "Miners' Refuge")
                    source_lines.append(f'• [{doc_title}]({doc_url}) — {source_label}')
                if len(source_lines) >= 8:
                    break
            if source_lines:
                sources_value = '\n'.join(source_lines)

        # Build paginated embeds -------------------------------------------------
        embed_title = title or f'❓ {question}'
        pages = split_response(answer)
        total = len(pages)
        footer_base = (
            f"Documentação • {DOCS_BASE_URL} • "
            "💬 Responda a esta mensagem para continuar a conversa"
        )
        embeds: list[discord.Embed] = []
        for i, page_text in enumerate(pages):
            e = discord.Embed(
                title=embed_title if i == 0 else '',
                description=page_text,
                color=discord.Color.orange() if spark_report is not None else discord.Color.blue(),
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

    def _store_conversation(
        self,
        message_id: int,
        question: str,
        answer: str,
        spark_report: SparkReport | None = None,
    ) -> None:
        """Store a conversation exchange for follow-up replies."""
        self._conversations[message_id] = {
            'question': question,
            'answer': answer,
            'history': [],
            'spark_report': spark_report,
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

        # Enforce per-user cooldown on follow-up replies
        user_id = message.author.id
        if user_id in self._followup_cd:
            return
        self._followup_cd[user_id] = True

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

        # Carry the Spark report forward through the conversation chain.
        spark_report: SparkReport | None = conv.get('spark_report')

        async with message.channel.typing():
            try:
                history = conv.get('history', []).copy()
                history.append({'question': conv['question'], 'answer': conv['answer']})

                answer, embeds = await self._run_agent(
                    follow_up_question,
                    history=history,
                    image_url=image_url,
                    spark_report=spark_report,
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
                    'spark_report': spark_report,
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

    # --- Public Spark integration point ---

    async def run_spark_analysis(
        self,
        interaction: discord.Interaction,
        report: SparkReport,
    ) -> None:
        """Fetch a Spark report summary from the LLM and store the conversation.

        Called by ``SparkAnalyzer`` after fetching and parsing the report.
        The interaction must already be deferred (``defer(thinking=True)``)
        before this is called.
        """
        server_tag = f'{report.platform_brand} {report.minecraft_version}'.strip()
        embed_title = f'🔥 Spark — {server_tag}' if server_tag else '🔥 Relatório Spark'
        question = (
            'Analise este relatório Spark e identifique os principais problemas de '
            'desempenho, se houver. Siga o protocolo diagnóstico.'
        )
        try:
            answer, embeds = await self._run_agent(
                question,
                spark_report=report,
                title=embed_title,
            )
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            self._store_conversation(msg.id, question, answer, spark_report=report)

        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in Spark analysis")
            await interaction.followup.send(
                'Ocorreu um erro ao analisar o relatório. Tente novamente mais tarde.'
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
