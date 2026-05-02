"""Shared text utilities used across multiple cogs."""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import discord
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def run_tool_loop(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    exec_tool: Callable[[str, dict[str, Any]], Awaitable[tuple[str, list[dict]]]],
    status_label: Callable[[str, dict[str, Any]], str] | None = None,
    interaction: discord.Interaction | None = None,
    max_rounds: int = 6,
    dedup_key: Callable[[dict], str] | None = None,
) -> tuple[str, list[dict]]:
    """Run an agentic tool-calling loop until the LLM produces a final answer.

    Parameters
    ----------
    client:
        The OpenAI-compatible async client.
    model:
        The model ID to use for all chat completions.
    messages:
        The message list (system prompt, history, current question).  Mutated
        in place with tool calls and results.
    tools:
        The tool definitions to offer, or ``None`` to skip the loop entirely.
    exec_tool:
        Async callable ``(name, args) -> (result_text, sources)``.
    status_label:
        Optional callable ``(name, args) -> status_text`` shown on the
        deferred interaction while a tool runs.  When ``None``, the status is
        not updated.
    interaction:
        Optional Discord interaction to update with live status messages.
    max_rounds:
        Maximum iterations of the tool-calling loop.
    dedup_key:
        Optional callable ``(source_dict) -> str`` that returns a unique
        identifier for a source.  When provided, duplicate sources across
        rounds are filtered out.  When ``None``, no deduplication is applied.

    Returns
    -------
    ``(answer_text, all_sources)`` where *all_sources* is the deduplicated list
    of source dicts aggregated across all tool rounds.
    """
    all_sources: list[dict] = []
    seen_keys: set[str] = set()

    for _ in range(max_rounds):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            tools=tools,
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            break

        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if interaction is not None and status_label is not None:
                try:
                    await interaction.edit_original_response(
                        content=status_label(tc.function.name, args)
                    )
                except discord.HTTPException:
                    pass

            result_text, sources = await exec_tool(tc.function.name, args)

            if sources:
                if dedup_key is not None:
                    new_sources = [
                        s for s in sources if dedup_key(s) not in seen_keys
                    ]
                    if len(new_sources) < len(sources):
                        logger.info(
                            "Filtered %d duplicate source(s) from %s tool result",
                            len(sources) - len(new_sources),
                            tc.function.name,
                        )
                    for s in sources:
                        k = dedup_key(s)
                        if k:
                            seen_keys.add(k)
                    all_sources.extend(new_sources)

                    if not new_sources and sources:
                        result_text = (
                            'Os resultados desta busca já foram retornados '
                            'em uma rodada anterior. Use as informações já '
                            'fornecidas para formular sua resposta.'
                        )
                else:
                    all_sources.extend(sources)

            messages.append({
                'role': 'tool',
                'tool_call_id': tc.id,
                'content': truncate_safe(result_text, limit=6000),
            })
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
        )

    answer = response.choices[0].message.content or 'Não foi possível gerar uma resposta.'
    return answer, all_sources


def truncate_safe(text: str, limit: int = 3800, suffix: str = '\n\n...') -> str:
    """Truncate *text* at the last newline before *limit* to avoid breaking markdown."""
    if len(text) <= limit:
        return text
    idx = text.rfind('\n', 0, limit)
    if idx == -1:
        idx = limit
    return text[:idx] + suffix


def split_response(text: str, chunk_size: int = 3500) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* chars, breaking at newlines."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text)
            break
        idx = text.rfind('\n', 0, chunk_size)
        if idx == -1:
            idx = chunk_size
        chunks.append(text[:idx])
        text = text[idx:].lstrip('\n')
    return chunks


def build_source_pages(
    source_lines: list[str],
    title: str,
    color: discord.Color,
    footer_base: str,
    page_size: int = 4000,
) -> list[discord.Embed]:
    """Paginate *source_lines* into embeds using the description field (4096 limit).

    Each embed's description stays under *page_size* chars.  Sources are
    always placed on their own page(s) so they never hit the 1024-char
    field-value limit.
    """
    if not source_lines:
        return []

    pages: list[discord.Embed] = []
    current_lines: list[str] = []
    current_len = 0

    for line in source_lines:
        added_len = len(line) + (1 if current_lines else 0)
        if current_lines and current_len + added_len > page_size:
            e = discord.Embed(
                title=title,
                description='\n'.join(current_lines),
                color=color,
            )
            pages.append(e)
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += added_len

    if current_lines:
        e = discord.Embed(
            title=title,
            description='\n'.join(current_lines),
            color=color,
        )
        pages.append(e)

    total = len(pages) + 0  # pages of sources only
    for i, e in enumerate(pages):
        src_idx = i + 1
        e.set_footer(text=f"Fontes {src_idx}/{len(pages)} • {footer_base}")

    return pages


class PaginatedEmbedView(discord.ui.View):
    """Navigation view for multi-page embed responses."""

    def __init__(self, embeds: list[discord.Embed]):
        super().__init__(timeout=120)
        self.embeds = embeds
        self.index = 0
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self.prev_btn.disabled = self.index == 0
        self.next_btn.disabled = self.index >= len(self.embeds) - 1
        self.counter_btn.label = f'{self.index + 1}/{len(self.embeds)}'

    @discord.ui.button(label='◄', style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index -= 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)

    @discord.ui.button(label='1/1', style=discord.ButtonStyle.secondary, disabled=True)
    async def counter_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()

    @discord.ui.button(label='►', style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index += 1
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.index], view=self)
