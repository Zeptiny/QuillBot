import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import time

import aiohttp
import discord
import numpy as np
from cachetools import TTLCache
from discord import app_commands
from discord.ext import commands, tasks
from openai import AsyncOpenAI, RateLimitError

from cogs.conversation_store import (
    ConversationStore as _ConversationStore,
    add_participant as _add_participant,
    author_info as _author_info,
    build_conversation_block as _build_conversation_block,
    build_current_message as _build_current_message,
    build_history_messages as _build_history_messages,
    cap_turns as _cap_turns,
    make_turn as _make_turn,
)
from cogs.memory import MEMORY_ABOUT_TOOL, MEMORY_SEARCH_TOOL, MEMORY_WRITE_TOOL
from cogs.plugin_apis import HTTP_HEADERS as _HTTP_HEADERS
from cogs.plugin_apis import search_all as _search_plugins_all
from cogs.spark_parser import (
    AVAILABLE_SECTIONS as _SPARK_SECTIONS,
    SparkReport,
    build_detail as _spark_build_detail,
    build_summary as _spark_build_summary,
)
from cogs.utils import PaginatedEmbedView, build_source_pages, run_tool_loop, split_response
from cogs.utils import (
    CHANNEL_HISTORY_TOOL as _CHANNEL_HISTORY_TOOL,
    COUNT_MENTIONS_TOOL as _COUNT_MENTIONS_TOOL,
    GET_MESSAGE_CONTEXT_TOOL as _GET_MESSAGE_CONTEXT_TOOL,
    GET_USER_STATS_TOOL as _GET_USER_STATS_TOOL,
    GUILD_INFO_TOOL as _GUILD_INFO_TOOL,
    SEARCH_HISTORY_TOOL as _SEARCH_HISTORY_TOOL,
    build_full_context_block as _build_full_context_block,
    build_guild_context as _build_guild_context,
    fetch_channel_history as _fetch_channel_history,
    fetch_message_context as _fetch_message_context,
    fetch_recent_channel_context as _fetch_recent_channel_context,
)
from config import (
    CHANNEL_CONTEXT_MESSAGES,
    CHAT_MODEL,
    COOLDOWN_PER,
    COOLDOWN_RATE,
    CONVERSATIONS_DB_PATH,
    CONVERSATIONS_HISTORY_TURNS,
    CONVERSATIONS_MAX_STORED,
    CONVERSATIONS_MAX_TURNS,
    CONVERSATIONS_TTL_SECONDS,
    DOC_SOURCES,
    DOCS_BASE_URL,
    DOCS_BRANCH,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GITHUB_API,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_MODEL,
    MEMORY_ENABLED,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    REINDEX_INTERVAL_HOURS,
    RERANK_AVAILABLE,
    RERANK_MODEL,
    SPARK_MODEL,
    VECTOR_STORE_PATH,
)

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6  # Safety cap on tool-calling iterations


def _conversation_participants(message: discord.Message) -> set[str]:
    """User ids relevant to a message: author, mentions and reply target."""
    ids = {str(message.author.id)}
    for u in getattr(message, 'mentions', []):
        if not u.bot:
            ids.add(str(u.id))
    ref = getattr(message, 'reference', None)
    ref_msg = getattr(ref, 'resolved', None)
    ref_author = getattr(ref_msg, 'author', None)
    if ref_author is not None and not ref_author.bot:
        ids.add(str(ref_author.id))
    return ids

_MEMORY_INSTRUCTIONS = (
    "- Você tem uma memória persistente: um bloco <memory> com lembranças relevantes é "
    "injetado automaticamente no início de cada resposta. Use-o naturalmente — nunca "
    "mencione o bloco nem os IDs [mem #N]. Para lembrar de algo não injetado, use "
    "`memory_search`; para contextualizar quem é alguém, use `memory_about`.\n"
    "- Ao descobrir algo digno de memória (preferência estável, fato sobre pessoa, evento "
    "marcante, habilidade ensinada), salve com `memory_write` — uma frase autocontida por "
    "memória, em terceira pessoa. Atualize com action=update quando o fato mudar e use "
    "action=forget quando deixar de valer. Use pinned=true só para fatos centrais e "
    "permanentes.\n"
) if MEMORY_ENABLED else ""

_DISCORD_FORMAT_GUIDE = (
    "A resposta será exibida no Discord (embed description) — use APENAS sintaxe que o Discord renderiza:\n"
    "- Permitido: **negrito**, *itálico*, __sublinhado__, ~~tachado~~, `código inline`, "
    "```bloco de código``` com linguagem (yaml, properties, json, toml, bash, log), "
    "> citação, - lista, 1. lista numerada, ||spoiler||, ### título / ## título, [texto](url).\n"
    "- Proibido: tabelas markdown com |, separadores --- ou ***, HTML (<br>, <div>), LaTeX ($$), footnotes.\n"
    "- Para comparações use listas com **Chave**: valor — NUNCA tabelas com pipes.\n"
    "- Para títulos de seção use ### Título ou **Negrito** — nunca ---.\n"
    "- Blocos de código sempre com linguagem para highlight; valores de config em ```yaml ou ```properties.\n"
    "- Cite arquivos/chaves com `inline code` (ex: `paper.yml`, `view-distance`).\n"
)

SYSTEM_PROMPT = (
    "<role>\n"
    "Você é o assistente oficial do Miners' Refuge, uma comunidade brasileira de "
    "administradores de servidores Minecraft. Responda sempre em português brasileiro.\n"
    "</role>\n\n"
    "<instructions>\n"
    "- Para qualquer pergunta técnica sobre configuração, administração ou otimização de "
    "servidores Minecraft, chame `search_docs` imediatamente.\n"
    + _MEMORY_INSTRUCTIONS +
    "- Baseie cada resposta nos dados retornados pelas ferramentas, não em conhecimento de treinamento.\n"
    "- Se a pergunta for vaga ou ambígua, peça esclarecimentos — omita chamadas de ferramentas.\n"
    "- Cite valores e trechos de configuração exatamente como retornados pelas ferramentas. "
    f"Se nenhuma ferramenta retornar dados relevantes, diga que não encontrou e sugira visitar {DOCS_BASE_URL}.\n"
    "- Omita seções de fontes na resposta — as fontes são exibidas automaticamente pela interface.\n"
    "- Execute as ferramentas diretamente sem pedir autorização. "
    "Se múltiplas buscas independentes forem necessárias, execute-as em paralelo na mesma rodada.\n"
    "</instructions>\n\n"
    "<response_format>\n"
    + _DISCORD_FORMAT_GUIDE
    + "</response_format>\n\n"
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
    "Para servidores com 20–50 jogadores, valores entre 6 e 8 oferecem bom equilíbrio.\n"
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
    "6. **Hotspots** — Chame `get_spark_detail(\"hotspots\")` para a árvore completa. "
    "Identifique os métodos com maior `self_pct` e cite o percentual exato de cada um "
    "(ex: `EntityTick` → **73,2% self_pct**). Traduza o stack em causa raiz legível: "
    "qual plugin, sistema ou entidade está consumindo CPU e exatamente quanto (%).\n"
    "7. **Configuração** — Use `get_config_key(file, key)` para verificar o valor "
    "atual da configuração suspeita.\n"
    "8. **Recomendação fundamentada** — Chame `search_docs` para cada causa identificada, "
    "independentemente de o servidor estar com lag ou não. "
    "Se há problemas: busque valores recomendados (ex: 'entity activation range paper', 'villager lag'). "
    "Se o servidor está saudável: confirme boas práticas (ex: 'paper performance best practices', 'view-distance recommendations'). "
    "Nunca invente valores de configuração — toda recomendação deve vir dos resultados de `search_docs`.\n"
    "</diagnostic_protocol>\n\n"
    "<platform_awareness>\n"
    "Antes de recomendar qualquer configuração, identifique a plataforma do servidor a partir do campo "
    "'Software' no resumo do relatório:\n"
    "- **Paper / Purpur / Folia**: configs paper.yml, spigot.yml, bukkit.yml, paper-world-defaults.yml existem.\n"
    "- **Spigot / CraftBukkit**: spigot.yml e bukkit.yml existem; paper.yml NÃO existe.\n"
    "- **Forge / NeoForge**: paper.yml, spigot.yml e bukkit.yml NÃO existem. "
    "Use server.properties + configs de mods. Ao chamar `search_docs`, inclua o nome da plataforma na query "
    "(ex: 'neoforge entity lag optimization', 'forge server performance').\n"
    "- **Fabric / Quilt**: similar ao Forge — sem spigot.yml. Busque por mods de otimização (Lithium, Krypton).\n"
    "- **Vanilla**: apenas server.properties.\n"
    "NUNCA recomende arquivos de configuração que não existem para a plataforma identificada.\n"
    "</platform_awareness>\n\n"
    "<tool_guidance>\n"
    "Use as ferramentas proativamente e na ordem indicada acima.\n"
    "Prefira `get_config_key` quando precisar de apenas uma chave. "
    "Use `get_spark_detail` para dados completos de uma seção.\n"
    "Sempre baseie recomendações de configuração em `search_docs`, não em conhecimento de treinamento. "
    "Para cada problema identificado nos hotspots, chame `search_docs` com uma query específica "
    "(ex: 'villager lag optimization paper', 'entity activation range') — nunca invente valores.\n"
    "</tool_guidance>\n\n"
    "<spark_response_format>\n"
    + _DISCORD_FORMAT_GUIDE
    + "Regras adicionais para Spark:\n"
    "- Liste métricas com **Negrito**: valor (ex: **TPS**: 18.5, **MSPT mediana**: 62 ms).\n"
    "- Agrupe por seção com ### Título ou **Negrito** (ex: ### Diagnóstico).\n"
    "- Ao citar hotspots, sempre inclua o percentual exato: `NomeDoMétodo` — **XX,X% self_pct**.\n"
    "- Para cada problema: causa raiz + percentual de impacto + configuração atual + valor recomendado (da documentação).\n"
    "- Valores de configuração sempre em bloco ```yaml / ```properties com linguagem.\n"
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
                            'Número máximo de resultados (padrão: 5, intervalo: 1-12). '
                            'Use 3-5 para perguntas focadas, 8-12 para tópicos amplos.'
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
if MEMORY_ENABLED:
    TOOLS.extend([MEMORY_SEARCH_TOOL, MEMORY_WRITE_TOOL, MEMORY_ABOUT_TOOL])
TOOLS.extend([
    _CHANNEL_HISTORY_TOOL,
    _GUILD_INFO_TOOL,
    _SEARCH_HISTORY_TOOL,
    _GET_MESSAGE_CONTEXT_TOOL,
    _GET_USER_STATS_TOOL,
    _COUNT_MENTIONS_TOOL,
])

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
        api_key = OPENAI_API_KEY or "not-needed"
        try:
            self.client = AsyncOpenAI(
                base_url=OPENAI_BASE_URL,
                api_key=api_key,
            )
            if not OPENAI_API_KEY and 'openrouter.ai' in OPENAI_BASE_URL:
                logger.warning("OPENAI_API_KEY/OPENROUTER_API_KEY not set -- LLM calls will fail until configured")
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            self.client = None
        self._local_embed_model = None
        self._local_embed_dim: int | None = None
        self.session: aiohttp.ClientSession | None = None
        self.chunks: list[dict] = []  # {content, path, title, embedding, source, doc_url}
        self._last_commit_sha: str | None = None
        self._indexing: bool = False
        self._emb_matrix: np.ndarray | None = None  # (N, dim) float32 for vectorized search
        # Per-source SHA tracking for granular reindex
        self._source_shas: dict[str, str] = {}  # label -> latest commit SHA
        self._source_last_index: dict[str, float] = {}  # label -> timestamp
        # Persistent conversations with activity-based TTL (survive restarts)
        self.store = _ConversationStore(
            CONVERSATIONS_DB_PATH,
            kind='ask',
            ttl_seconds=CONVERSATIONS_TTL_SECONDS,
            max_stored=CONVERSATIONS_MAX_STORED,
        )
        # Spark reports are too heavy to persist — kept in memory per conversation
        self._spark_by_conv: dict[str, SparkReport] = {}
        # Per-user follow-up cooldown (same period as slash commands)
        self._followup_cd: TTLCache = TTLCache(maxsize=500, ttl=COOLDOWN_PER)

    def _get_local_model(self):
        if self._local_embed_model is not None:
            return self._local_embed_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s (device=%s)", LOCAL_EMBEDDING_MODEL, LOCAL_EMBEDDING_DEVICE)
            self._local_embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=LOCAL_EMBEDDING_DEVICE)
            test_emb = self._local_embed_model.encode(["test"], normalize_embeddings=True)
            self._local_embed_dim = test_emb.shape[1] if hasattr(test_emb, 'shape') else len(test_emb[0])
            logger.info("Local embedding model loaded (dim=%s)", self._local_embed_dim)
        except ImportError:
            logger.error("sentence-transformers not installed -- install with pip install sentence-transformers to use EMBEDDING_PROVIDER=local")
            raise
        except Exception:
            logger.exception("Failed to load local embedding model %s", LOCAL_EMBEDDING_MODEL)
            raise
        return self._local_embed_model


    async def cog_load(self):
        self.session = aiohttp.ClientSession(
            headers=_HTTP_HEADERS,
            timeout=aiohttp.ClientTimeout(total=15),
        )
        if EMBEDDING_PROVIDER == 'local':
            try:
                await asyncio.to_thread(self._get_local_model)
            except Exception:
                logger.warning("Falling back to remote embeddings due to local model load failure")
        loaded = await asyncio.to_thread(self._load_vectors)
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
                if EMBEDDING_PROVIDER == 'local' and self._local_embed_dim and embeddings.shape[1] != self._local_embed_dim:
                    logger.warning(
                        "Embedding dim mismatch (stored %d vs local model %d), will reindex",
                        embeddings.shape[1], self._local_embed_dim,
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
        if EMBEDDING_PROVIDER == 'local':
            model = self._get_local_model()
            embeddings = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, show_progress_bar=False)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return [list(e) for e in embeddings]
        if not self.client:
            raise RuntimeError("OpenAI client not initialized -- check OPENAI_API_KEY / OPENAI_BASE_URL")
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

    async def index_docs(self, sources: list[dict] | None = None):
        """Fetch docs from all configured sources (or only *sources* if provided) and create embeddings."""
        if self._indexing:
            logger.info("index_docs() called while already indexing; skipping")
            return
        self._indexing = True
        try:
            await self._index_docs_inner(sources)
        finally:
            self._indexing = False

    async def _index_docs_inner(self, sources: list[dict] | None = None):
        sources_to_index = sources if sources is not None else DOC_SOURCES
        logger.info(
            "Indexing %d documentation source(s)...",
            len(sources_to_index),
        )

        # All sources share a semaphore to cap total concurrent HTTP fetches
        semaphore = asyncio.Semaphore(5)
        source_tasks = [self._index_source(src, semaphore) for src in sources_to_index]
        source_results = await asyncio.gather(*source_tasks, return_exceptions=True)

        all_chunks = []
        indexed_labels: list[str] = []
        for src, result in zip(sources_to_index, source_results, strict=True):
            if isinstance(result, Exception):
                logger.error("Error indexing source '%s': %s", src['label'], result)
            else:
                all_chunks.extend(result)
                indexed_labels.append(src['label'])

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

        # Track per-source SHAs and timestamps
        if sources is not None:
            # Partial reindex: update only the requested sources
            for src in sources_to_index:
                label = src['label']
                url = f'https://api.github.com/repos/{src["repo"]}/commits/{src["branch"]}'
                try:
                    async with self.session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._source_shas[label] = data.get('sha', '')
                except Exception:
                    logger.exception("Failed to fetch commit SHA for %s", label)
                self._source_last_index[label] = __import__('time').monotonic()
        else:
            # Full reindex: update all sources and composite SHA
            self._last_commit_sha = await self._get_composite_sha()
            for src in DOC_SOURCES:
                label = src['label']
                sha = self._source_shas.get(label, '')
                if not sha:
                    url = f'https://api.github.com/repos/{src["repo"]}/commits/{src["branch"]}'
                    try:
                        async with self.session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                sha = data.get('sha', '')
                                self._source_shas[label] = sha
                    except Exception:
                        logger.exception("Failed to fetch commit SHA for %s", label)
                self._source_last_index[label] = __import__('time').monotonic()

        await asyncio.to_thread(self._save_vectors)

    # --- Search with Reranking ---

    async def _rerank(self, query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
        """Rerank documents using the reranking model via OpenAI-compatible API."""
        if not RERANK_AVAILABLE:
            return documents[:top_n]
        try:
            headers = {
                'Authorization': f'Bearer {OPENAI_API_KEY}',
                'Content-Type': 'application/json',
            }
            payload = {
                'model': RERANK_MODEL,
                'query': query,
                'documents': [d['content'] for d in documents],
                'top_n': top_n,
            }
            async with self.session.post(
                f'{OPENAI_BASE_URL}/rerank',
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
            await interaction.edit_original_response(content='💭 Pensando…')
        except discord.HTTPException:
            pass

        logger.info(
            "Processing /ask user=%s guild=%s question=%r",
            interaction.user.id, interaction.guild_id, question[:80],
        )

        try:
            answer, embeds, sources = await self._run_agent(
                question, image_url=image_url, interaction=interaction
            )
            # Clear any in-progress status message before sending the final embed.
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            await self._store_conversation(
                msg, question, answer, sources=sources, interaction=interaction
            )

        except RateLimitError:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in /ask command")
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(
                'Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.'
            )

    # --- Tool execution ---

    async def _exec_tool(
        self,
        name: str,
        args: dict,
        spark_report: SparkReport | None = None,
        bot: discord.Client | None = None,
        channel: discord.abc.Messageable | None = None,
        guild: discord.Guild | None = None,
        user: discord.abc.User | discord.Member | None = None,
        origin: str | None = None,
        participant_ids: set[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """Execute a tool call and return (result_text, source_chunks)."""
        if name in ('memory_search', 'memory_write', 'memory_about'):
            mem_cog = self.bot.get_cog('Memory')
            if not mem_cog:
                return 'Memória não disponível.', []
            if not guild:
                return 'Memória requer estar em um servidor.', []
            actor_name = f'bot (via {user.display_name})' if user is not None else 'bot'
            return await mem_cog.exec_tool(
                name, args, guild=guild, actor_name=actor_name,
                requester=user, channel=channel,
                origin=origin,
                participant_ids=participant_ids,
            )

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

        if name == 'get_channel_history':
            b = bot or self.bot
            ch = channel
            limit = args.get('limit', 20)
            cid = args.get('channel_id')
            text = await _fetch_channel_history(b, ch, limit=limit, channel_id=cid)
            return text, []

        if name == 'get_guild_info':
            g = guild
            if not g:
                return "Fora de um servidor (DM).", []
            text = _build_guild_context(g)
            try:
                chs = [f"#{c.name} ({c.id})" for c in g.channels if isinstance(c, discord.TextChannel)][:30]
                if chs:
                    text += "\nCanais de texto: " + ", ".join(chs)
            except Exception:
                pass
            return text, []

        if name == 'search_history':
            hist = self.bot.get_cog('HistoryRAG')
            if not hist:
                return "Histórico não disponível.", []
            g = guild
            if not g:
                return "Busca no histórico requer estar em um servidor.", []
            query = args.get('query', '')
            limit = max(1, min(12, int(args.get('limit', 5))))
            try:
                results = await hist.search(query, g.id, limit=limit, channel_id=args.get('channel_id'), author_id=args.get('author_id'), author_name=args.get('author_name'), after=args.get('after'), before=args.get('before'), search_mode=args.get('search_mode','hybrid'), sort_by=args.get('sort_by','relevance'))  # type: ignore
            except Exception:
                logger.exception("search_history failed in docs_rag")
                return "Erro ao buscar no histórico.", []
            if not results:
                return "Nenhuma mensagem relevante encontrada no histórico.", []
            parts = []
            for r in results:
                jump = r.get('jump_url', '')
                link = f"[ver]({jump})" if jump else ""
                window = r.get('chunk_text', r.get('content', ''))[:1200]
                header = f"**{r.get('author_full','?')}** em #{r.get('channel_name','?')} — {r.get('ts','')} {link} (score {r.get('_score',0):.2f})"
                parts.append(f"{header}\n```\n{window}\n```\n`msg_id={r.get('msg_id')} channel_id={r.get('channel_id')}`")
            return "\n\n---\n\n".join(parts), []

        if name == 'get_user_stats':
            hist = self.bot.get_cog('HistoryRAG')
            if not hist:
                return "Histórico não disponível.", []
            g = guild
            if not g:
                return "Requer servidor.", []
            try:
                stats = await hist.get_user_stats(g.id, author_id=args.get('author_id'), author_name=args.get('author_name'))  # type: ignore
            except Exception:
                logger.exception("get_user_stats failed")
                return "Erro ao buscar estatísticas.", []
            if "error" in stats:
                return stats["error"], []
            lines = [f"Usuário: {stats['author_full']} ({', '.join(stats['author_ids'])})", f"Total: {stats['total_messages']} msgs | Média: {stats['avg_length']} chars", f"Canais: {', '.join(f'{k}={v}' for k,v in stats['top_channels'])}", f"Horários: {', '.join(f'{h}h={v}' for h,v in stats['top_hours'])}", f"Período: {stats['first_seen']} → {stats['last_seen']}", f"Exemplo: {stats['example_content']} {stats['example_jump']}"]
            return "\n".join(lines), []

        if name == 'count_mentions':
            hist = self.bot.get_cog('HistoryRAG')
            if not hist:
                return "Histórico não disponível.", []
            g = guild
            if not g:
                return "Requer servidor.", []
            try:
                groups = await hist.count_mentions(g.id, query=args.get('query',''), group_by=args.get('group_by','author'), limit=int(args.get('limit',10)), after=args.get('after'), before=args.get('before'))  # type: ignore
            except Exception:
                logger.exception("count_mentions failed")
                return "Erro ao contar menções.", []
            if not groups:
                return "Nenhuma menção encontrada.", []
            lines = [f"{gr['key']}: {gr['count']}× — ex: {gr['example'].get('content','')[:120]}" for gr in groups]
            return "\n".join(lines), []

        if name == 'get_message_context':
            text = await _fetch_message_context(self.bot, channel_id=args.get('channel_id',''), message_id=args.get('message_id',''), window=args.get('window', 5))
            return text, []

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

    # --- Status helpers ---

    @staticmethod
    def _status_label(tool_name: str, args: dict) -> str:
        """Return a short Portuguese status string shown while a tool runs."""
        if tool_name == 'search_docs':
            query = args.get('query', '')
            source = args.get('source')
            label = f'🔍 Pesquisando documentação: *{query[:60]}*'
            if source:
                label += f' ({source})'
            return label
        if tool_name == 'search_plugins':
            query = args.get('query', '')
            return f'🔌 Pesquisando plugins: *{query[:60]}*'
        if tool_name == 'get_channel_history':
            lim = args.get('limit', 20)
            cid = args.get('channel_id')
            return f'📜 Lendo histórico ({lim} msgs)' + (f' canal {cid}' if cid else '')
        if tool_name == 'get_guild_info':
            return '🏰 Coletando informações do servidor'
        if tool_name == 'search_history':
            q = args.get('query','')[:40]
            return f'🔎 Buscando no histórico: *{q}*'
        if tool_name == 'get_user_stats':
            return f'📊 Estatísticas de {args.get("author_id") or args.get("author_name","usuário")}…'
        if tool_name == 'count_mentions':
            return f'🔢 Contando menções: *{args.get("query","")[:30]}*'
        if tool_name == 'get_message_context':
            return f'🧩 Contexto da mensagem {args.get("message_id","")}…'
        if tool_name == 'memory_search':
            return f"🧠 Recordando: *{args.get('query', '')[:40]}*"
        if tool_name == 'memory_write':
            act = args.get('action', 'write')
            tgt = str(args.get('memory_id') or args.get('content') or args.get('content_match') or '')[:40]
            return f'🧠 Memória — {act}: *{tgt}*'
        if tool_name == 'memory_about':
            return f"🧠 Relembrando {args.get('user', 'quem pergunta')}…"
        if tool_name == 'get_spark_detail':
            section = args.get('section', '')
            section_names = {
                'hotspots': 'árvore de hotspots CPU',
                'jvm': 'flags JVM e GC',
                'profiler': 'estatísticas TPS/MSPT',
                'configs': 'arquivos de configuração',
                'game_rules': 'game rules',
                'world': 'dados de mundo',
                'plugins': 'plugins',
            }
            if section.startswith('configs:'):
                file = section[len('configs:'):]
                return f'📂 Lendo configuração: `{file}`'
            readable = section_names.get(section, section)
            return f'📊 Analisando relatório Spark — {readable}'
        if tool_name == 'get_config_key':
            file = args.get('file', '')
            key = args.get('key', '')
            return f'⚙️ Verificando `{key}` em `{file}`'
        return f'🔧 Executando: {tool_name}'

    # --- Agentic loop ---

    async def _run_agent(
        self,
        question: str,
        history: list[dict] | None = None,
        image_url: str | None = None,
        image_urls: list[str] | None = None,
        spark_report: SparkReport | None = None,
        reply_to: str | None = None,
        title: str | None = None,
        interaction: discord.Interaction | None = None,
        user: discord.abc.User | discord.Member | None = None,
        guild: discord.Guild | None = None,
        channel: discord.abc.Messageable | None = None,
        created_at: datetime.datetime | None = None,
        participant_ids: set[str] | None = None,
        origin: str | None = None,
        context_message: discord.Message | None = None,
    ) -> tuple[str, list[discord.Embed], list[dict]]:
        """Run the LLM with tool-calling in a loop until it produces a final answer.

        When ``spark_report`` is provided the agent uses ``SPARK_MODEL``,
        receives the report summary as a synthetic prior exchange, and gains
        access to the ``get_spark_detail`` / ``get_config_key`` Spark tools.
        """
        # Resolve context from interaction if not explicitly passed
        if interaction is not None:
            user = user or interaction.user
            guild = guild or interaction.guild
            channel = channel or interaction.channel
            created_at = created_at or interaction.created_at
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
        history = history or []
        if history:
            system_content += '\n\n' + _build_conversation_block(
                history[-CONVERSATIONS_HISTORY_TURNS:],
                current_author=_author_info(user or (interaction.user if interaction else None)),
            )

        # Inject persistent memory (pinned + semantically selected) ----------------
        if MEMORY_ENABLED and guild is not None:
            mem_cog = self.bot.get_cog('Memory')
            if mem_cog is not None:
                try:
                    mem_block = await mem_cog.build_memory_block(
                        guild.id, question,
                        speaker_id=str(user.id) if user is not None else None,
                        participant_ids=participant_ids,
                    )
                    if mem_block:
                        system_content += f'\n\n{mem_block}'
                except Exception:
                    logger.exception('Failed to build memory block for _run_agent')

        # Recent channel messages as conversation context ------------------------
        if CHANNEL_CONTEXT_MESSAGES > 0:
            try:
                chan_ctx = await _fetch_recent_channel_context(
                    self.bot, channel, before=context_message,
                )
                if chan_ctx:
                    system_content += f'\n\n{chan_ctx}'
            except Exception:
                logger.exception('Failed to build recent channel context for _run_agent')

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

        # Inject user/guild/channel/temporal awareness ---------------------------
        if user or guild or channel or created_at:
            try:
                ctx_block = _build_full_context_block(user, guild, channel, created_at)
                messages.append({'role': 'user', 'content': ctx_block})
            except Exception:
                logger.exception("Failed to build context block for _run_agent")

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
            messages.extend(
                _build_history_messages(history, max_turns=CONVERSATIONS_HISTORY_TURNS)
            )

        # Current user message ---------------------------------------------------
        urls: list[str] = []
        if image_urls:
            urls.extend([u for u in image_urls if u])
        elif image_url:
            urls.append(image_url)
        messages.append(_build_current_message(
            question,
            author=_author_info(user or (interaction.user if interaction else None)),
            ts=created_at.timestamp() if created_at else None,
            image_urls=urls,
            reply_to=reply_to,
            in_conversation=bool(history),
        ))

        # Choose model and tool set based on session type ------------------------
        model = SPARK_MODEL if spark_report is not None else CHAT_MODEL
        active_tools = TOOLS + SPARK_TOOLS if spark_report is not None else TOOLS

        exec_tool = (
            lambda name, args: self._exec_tool(
                name, args, spark_report=spark_report, bot=self.bot, channel=channel,
                guild=guild, user=user, origin=origin, participant_ids=participant_ids,
            )
        )

        answer, all_sources = await run_tool_loop(
            client=self.client,
            model=model,
            messages=messages,
            tools=active_tools,
            exec_tool=exec_tool,
            status_label=self._status_label,
            interaction=interaction,
            max_rounds=MAX_TOOL_ROUNDS,
            dedup_key=lambda s: s.get('path', ''),
        )

        # Build source links (attached to first page only) -----------------------
        source_lines: list[str] = []
        sources: list[dict] = []
        if all_sources:
            seen_paths: set[str] = set()
            for r in all_sources:
                if r['path'] not in seen_paths:
                    seen_paths.add(r['path'])
                    doc_url = r.get('doc_url', path_to_docs_url(r['path']))
                    doc_title = r['title'] or _title_from_path(r['path'])
                    source_label = r.get('source', "Miners' Refuge")
                    source_lines.append(f'• [{doc_title}]({doc_url}) — {source_label}')
                    sources.append({'title': str(doc_title)[:100], 'url': doc_url})

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
            e.set_footer(
                text=f"Página {i + 1}/{total} • {footer_base}" if total > 1 else footer_base
            )
            embeds.append(e)

        if source_lines:
            embed_color = discord.Color.orange() if spark_report is not None else discord.Color.blue()
            embeds.extend(
                build_source_pages(
                    source_lines,
                    title='📄 Fontes da Documentação',
                    color=embed_color,
                    footer_base=footer_base,
                )
            )

        return answer, embeds, sources

    async def _store_conversation(
        self,
        message: discord.Message,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
        spark_report: SparkReport | None = None,
        interaction: discord.Interaction | None = None,
    ) -> None:
        """Persist a conversation exchange for follow-up replies."""
        user = interaction.user if interaction is not None else None
        guild = interaction.guild if interaction is not None else None
        channel = interaction.channel if interaction is not None else message.channel
        author = _author_info(user)
        ts_dt = interaction.created_at if interaction is not None else message.created_at
        ts = ts_dt.timestamp() if ts_dt else time.time()
        turn = _make_turn(
            question, answer,
            author=author, ts=ts,
            channel_id=getattr(channel, 'id', None),
            channel_name=getattr(channel, 'name', None),
            images=[],
            sources=sources,
        )
        origin = {
            'channel_id': str(getattr(channel, 'id', '') or ''),
            'channel_name': getattr(channel, 'name', '') or '',
            'guild_id': str(getattr(guild, 'id', '') or ''),
            'guild_name': getattr(guild, 'name', '') or '',
        }
        await self.store.create(
            str(message.id),
            guild_id=origin['guild_id'] or None,
            channel_id=origin['channel_id'] or None,
            data={
                'turns': [turn],
                'participants': [author] if author.get('id') else [],
                'origin': origin,
                'started_ts': ts,
            },
        )
        if spark_report is not None:
            self._spark_by_conv[str(message.id)] = spark_report

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle reply-based follow-up conversations."""
        if message.author.bot:
            return
        if not message.reference or not message.reference.message_id:
            return

        ref_id = message.reference.message_id
        conv = await self.store.get_by_handle(ref_id)
        if not conv:
            return

        # Enforce per-user cooldown on follow-up replies
        user_id = message.author.id
        if user_id in self._followup_cd:
            try:
                await message.reply('⏳ Aguarde antes de enviar outra resposta.', delete_after=5)
            except discord.HTTPException:
                pass
            return
        self._followup_cd[user_id] = True

        follow_up_question = message.content.strip()
        if self.bot.user:
            follow_up_question = re.sub(rf'<@!?{self.bot.user.id}>', '', follow_up_question)
            follow_up_question = re.sub(r'\s+', ' ', follow_up_question).strip()
        if not follow_up_question and not message.attachments:
            return

        image_urls: list[str] = [att.url for att in message.attachments if att.content_type and att.content_type.startswith('image/')]

        if not follow_up_question:
            follow_up_question = 'Analise esta imagem.'

        # Carry the Spark report forward through the conversation chain
        # (in-memory only — lost on restart, text history survives).
        spark_report: SparkReport | None = self._spark_by_conv.get(conv['conv_id'])

        async with message.channel.typing():
            try:
                history = conv['data'].get('turns', []).copy()

                answer, embeds, sources = await self._run_agent(
                    follow_up_question,
                    history=history,
                    image_urls=image_urls if image_urls else None,
                    spark_report=spark_report,
                    user=message.author,
                    guild=message.guild,
                    channel=message.channel,
                    created_at=message.created_at,
                    reply_to=str(ref_id),
                    participant_ids=_conversation_participants(message),
                    origin=message.jump_url,
                    context_message=message,
                )
                if len(embeds) == 1:
                    reply = await message.reply(embed=embeds[0])
                else:
                    reply = await message.reply(
                        embed=embeds[0], view=PaginatedEmbedView(embeds)
                    )

                turn = _make_turn(
                    follow_up_question, answer,
                    author=_author_info(message.author),
                    ts=message.created_at.timestamp(),
                    message_id=message.id,
                    channel_id=message.channel.id,
                    channel_name=getattr(message.channel, 'name', None),
                    images=image_urls,
                    sources=sources,
                    reply_to=ref_id,
                )
                data = conv['data']
                data['turns'] = _cap_turns(history + [turn], CONVERSATIONS_MAX_TURNS)
                _add_participant(data.setdefault('participants', []), _author_info(message.author))
                await self.store.update(
                    conv['conv_id'], data, new_handle_msg_id=reply.id,
                )
                if spark_report is not None:
                    self._spark_by_conv[conv['conv_id']] = spark_report
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
            answer, embeds, sources = await self._run_agent(
                question,
                spark_report=report,
                title=embed_title,
                interaction=interaction,
            )
            # Clear any in-progress status message before sending the final embed.
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            if len(embeds) == 1:
                msg = await interaction.followup.send(embed=embeds[0], wait=True)
            else:
                msg = await interaction.followup.send(
                    embed=embeds[0], view=PaginatedEmbedView(embeds), wait=True
                )
            await self._store_conversation(
                msg, question, answer,
                sources=sources, spark_report=report, interaction=interaction,
            )

        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in Spark analysis")
            await interaction.followup.send(
                'Ocorreu um erro ao analisar o relatório. Tente novamente mais tarde.'
            )

    @app_commands.command(
        name='reindex',
        description='Re-indexar a documentação (Admin)',
    )
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.describe(
        source='Fonte específica para reindexar (opcional; omitir para reindexar tudo)',
    )
    async def reindex(
        self,
        interaction: discord.Interaction,
        source: str | None = None,
    ):
        if self._indexing:
            await interaction.response.send_message(
                '📚 Já há uma indexação em andamento, aguarde a conclusão.',
                ephemeral=True,
            )
            return

        # Validate source name if provided
        if source:
            source_labels = [s['label'] for s in DOC_SOURCES]
            matching = [s for s in DOC_SOURCES if s['label'] == source]
            if not matching:
                await interaction.response.send_message(
                    f'Fonte "{source}" não encontrada. Fontes disponíveis: '
                    f'{", ".join(source_labels)}',
                    ephemeral=True,
                )
                return
            sources_to_index = matching
            description = f'Somente a fonte "{source}"'
        else:
            sources_to_index = None
            description = 'Toda a documentação'

        await interaction.response.defer(thinking=True)
        target_msg = f'📚 Reindexando {description}…' if description else '📚 Reindexando documentação…'
        try:
            await interaction.edit_original_response(content=target_msg)
        except discord.HTTPException:
            pass
        await self.index_docs(sources_to_index)
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(
            f'✅ {description} re-indexada! '
            f'({len(self.chunks)} chunks totais)'
        )


async def setup(bot: commands.Bot):
    if not OPENAI_API_KEY and 'openrouter.ai' in OPENAI_BASE_URL:
        logger.warning("OPENAI_API_KEY/OPENROUTER_API_KEY not set -- DocsRAG will use base_url %s with dummy key", OPENAI_BASE_URL)
    await bot.add_cog(DocsRAG(bot))
