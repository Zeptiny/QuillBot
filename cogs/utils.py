"""Shared text utilities used across multiple cogs."""

import discord


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
