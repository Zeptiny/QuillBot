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

from config import (
    CHAT_MODEL,
    COOLDOWN_PER,
    COOLDOWN_RATE,
    DOCS_BASE_URL,
    DOCS_BRANCH,
    DOCS_REPO,
    EMBEDDING_MODEL,
    GITHUB_API,
    GITHUB_RAW,
    OPENROUTER_API_KEY,
    REINDEX_INTERVAL_HOURS,
    RERANK_MODEL,
    VECTOR_STORE_PATH,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você é o assistente do Miners' Refuge, uma comunidade brasileira de administradores "
    "de servidores Minecraft. Responda em português brasileiro, de forma clara e concisa. "
    "Use APENAS as informações fornecidas no contexto da documentação abaixo. "
    "Se a resposta não estiver no contexto, diga que não encontrou na documentação "
    f"e sugira visitar {DOCS_BASE_URL}. "
    "NÃO inclua uma seção de fontes na sua resposta — as fontes são exibidas automaticamente."
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def path_to_docs_url(path: str) -> str:
    """Convert a repo file path to its docs website URL."""
    url = path.replace('.md', '').replace('README', '')
    url = url.rstrip('/')
    return f"{DOCS_BASE_URL}/{url}" if url else DOCS_BASE_URL


def _truncate_safe(text: str, limit: int = 3800) -> str:
    """Truncate text at the last newline before *limit* to avoid breaking markdown."""
    if len(text) <= limit:
        return text
    idx = text.rfind('\n', 0, limit)
    if idx == -1:
        idx = limit
    return text[:idx] + '\n\n...'


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
        self.chunks: list[dict] = []  # {content, path, title, embedding}
        self._last_commit_sha: str | None = None
        self._indexing: bool = False
        # TTL cache: max 200 conversations, each expires after 30 min
        self._conversations: TTLCache = TTLCache(maxsize=200, ttl=1800)

    async def cog_load(self):
        self.session = aiohttp.ClientSession()
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
            self.chunks = data.get('chunks', [])
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
        """Check for doc updates and reindex if the repo has new commits."""
        try:
            latest_sha = await self._get_latest_commit_sha()
            if latest_sha and latest_sha != self._last_commit_sha:
                logger.info(
                    "New commit detected (%s → %s), reindexing...",
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
        """Fetch the latest commit SHA from the docs repo."""
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
        # Exclude external URLs (http/https) — only match relative .md paths
        paths = re.findall(r'\((?!https?://)([^)]+\.md)\)', summary)
        return list(dict.fromkeys(paths))  # deduplicate preserving order

    def _extract_title(self, content: str) -> str:
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

    async def _fetch_doc(self, path: str, semaphore: asyncio.Semaphore) -> tuple[str, str | None]:
        """Fetch a single doc file, respecting the concurrency semaphore."""
        async with semaphore:
            content = await self._fetch(f'{GITHUB_RAW}/{path}')
        return path, content

    async def index_docs(self):
        """Fetch all docs from GitHub and create embeddings."""
        self._indexing = True
        try:
            await self._index_docs_inner()
        finally:
            self._indexing = False

    async def _index_docs_inner(self):
        logger.info("Indexing documentation from %s...", DOCS_REPO)

        summary = await self._fetch(f'{GITHUB_RAW}/SUMMARY.md')
        if not summary:
            logger.error("Failed to fetch SUMMARY.md — skipping indexing")
            return

        paths = self._extract_paths(summary)
        paths.insert(0, 'README.md')

        # Fetch documents concurrently (up to 5 at a time)
        semaphore = asyncio.Semaphore(5)
        tasks_list = [self._fetch_doc(p, semaphore) for p in paths]
        results = await asyncio.gather(*tasks_list)

        all_chunks = []
        fetched = 0
        for path, content in results:
            if content:
                chunks = self._chunk_text(content, path)
                all_chunks.extend(chunks)
                fetched += 1

        logger.info("Fetched %d/%d docs, created %d chunks", fetched, len(paths), len(all_chunks))

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

        # Save commit SHA and persist vectors
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
            results = await self.search(question)

            if not results:
                await interaction.followup.send(
                    'Não encontrei informações relevantes na documentação. '
                    f'Tente pesquisar diretamente em {DOCS_BASE_URL}'
                )
                return

            answer, embed = await self._build_answer(question, results, image_url=image_url)
            msg = await interaction.followup.send(embed=embed, wait=True)

            # Store conversation for follow-ups
            self._store_conversation(msg.id, question, answer, results)

        except RateLimitError:
            await interaction.followup.send(
                '⏳ Limite de requisições atingido. Tente novamente em alguns minutos.'
            )
        except Exception:
            logger.exception("Error in /ask command")
            await interaction.followup.send(
                'Ocorreu um erro ao processar sua pergunta. Tente novamente mais tarde.'
            )

    async def _build_answer(
        self,
        question: str,
        results: list[dict],
        history: list[dict] | None = None,
        image_url: str | None = None,
    ) -> tuple[str, discord.Embed]:
        """Generate an answer from search results and return (raw_answer, embed)."""
        context_parts = []
        for r in results:
            url = path_to_docs_url(r['path'])
            context_parts.append(
                f"[Fonte: {r['title']} — {url}]\n{r['content']}"
            )
        context = '\n\n---\n\n'.join(context_parts)

        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

        # Include conversation history for follow-ups
        if history:
            for h in history[-3:]:  # Last 3 exchanges max
                messages.append({'role': 'user', 'content': h['question']})
                messages.append({'role': 'assistant', 'content': h['answer']})

        user_text = f"Contexto da documentação:\n{context}\n\nPergunta: {question}"

        if image_url:
            # Vision format: content as list of parts
            messages.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_text},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            })
        else:
            messages.append({'role': 'user', 'content': user_text})

        response = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        answer = _truncate_safe(answer)

        embed = discord.Embed(
            title=f'❓ {question}',
            description=answer,
            color=discord.Color.blue(),
        )

        seen_paths = set()
        source_lines = []
        for r in results:
            if r['path'] not in seen_paths:
                seen_paths.add(r['path'])
                url = path_to_docs_url(r['path'])
                title = r['title'] or r['path']
                source_lines.append(f'• [{title}]({url})')
            if len(source_lines) >= 8:
                break

        embed.add_field(
            name='📄 Fontes da Documentação',
            value='\n'.join(source_lines),
            inline=False,
        )
        embed.set_footer(
            text=f'Miners\' Refuge Docs • {DOCS_BASE_URL} • 💬 Responda a esta mensagem para continuar a conversa'
        )

        return answer, embed

    def _store_conversation(self, message_id: int, question: str, answer: str, results: list[dict]):
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

        # User is replying to one of our /ask responses
        follow_up_question = message.content.strip()
        if not follow_up_question and not message.attachments:
            return

        # Check for image attachments
        image_url = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith('image/'):
                image_url = att.url
                break

        if not follow_up_question:
            follow_up_question = 'Analise esta imagem.'

        async with message.channel.typing():
            try:
                # Build history from previous exchanges
                history = conv.get('history', []).copy()
                history.append({'question': conv['question'], 'answer': conv['answer']})

                results = await self.search(follow_up_question)
                if not results:
                    await message.reply(
                        'Não encontrei informações relevantes na documentação. '
                        f'Tente pesquisar diretamente em {DOCS_BASE_URL}'
                    )
                    return

                answer, embed = await self._build_answer(
                    follow_up_question, results, history=history, image_url=image_url
                )
                reply = await message.reply(embed=embed)

                # Store new conversation entry so the chain can continue
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
        await interaction.response.defer(thinking=True)
        await self.index_docs()
        await interaction.followup.send(
            f'✅ Documentação re-indexada! ({len(self.chunks)} chunks)'
        )


async def setup(bot: commands.Bot):
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — DocsRAG cog will not be loaded")
        return
    await bot.add_cog(DocsRAG(bot))
