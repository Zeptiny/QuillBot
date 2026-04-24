"""SparkAnalyzer cog — Spark profiler report analysis for the Discord bot.

Responsibilities
----------------
- ``/spark <url>`` slash command: fetch report, hand off to DocsRAG for analysis.
- Passive URL detection: ``on_message`` watches for ``spark.lucko.me`` links and
  offers an "Analyze with AI" button via ``SparkAnalyzeView``.
- Manages its own ``aiohttp.ClientSession`` with a 30-second timeout (Spark
  JSON service can be slow on first fetch).
"""

from __future__ import annotations

import logging

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from cogs.spark_parser import SPARK_URL_PATTERN, SparkReport, fetch_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Button view — offered when a Spark URL is detected passively in chat
# ---------------------------------------------------------------------------

class SparkAnalyzeView(discord.ui.View):
    """One-time button that triggers AI analysis of a detected Spark report."""

    def __init__(self, cog: SparkAnalyzer, code: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.code = code
        self.message: discord.Message | None = None

    @discord.ui.button(
        label='Analisar com IA',
        style=discord.ButtonStyle.primary,
        emoji='🔥',
    )
    async def analyze(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await self.cog._do_spark_analysis(interaction, self.code, followup=True)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message is not None:
            try:
                await self.message.edit(view=self)
            except discord.HTTPException:
                pass


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class SparkAnalyzer(commands.Cog, name='SparkAnalyzer'):
    """Spark profiler report analysis commands and passive detection."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.session: aiohttp.ClientSession | None = None

    async def cog_load(self) -> None:
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def cog_unload(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    # --- /spark command ---

    @app_commands.command(
        name='spark',
        description='Analisa um relatório do Spark Profiler com IA',
    )
    @app_commands.describe(
        url='URL ou código do relatório (ex: https://spark.lucko.me/ABC ou apenas ABC)',
    )
    async def spark(self, interaction: discord.Interaction, url: str) -> None:
        await interaction.response.defer(thinking=True)
        await self._do_spark_analysis(interaction, url, followup=True)

    # --- Core analysis flow ---

    async def _do_spark_analysis(
        self,
        interaction: discord.Interaction,
        url_or_code: str,
        *,
        followup: bool = False,
    ) -> None:
        """Fetch the Spark report and delegate LLM analysis to DocsRAG.

        Parameters
        ----------
        interaction:
            The active Discord interaction (must already be deferred or
            responded to before calling with ``followup=True``).
        url_or_code:
            Full ``spark.lucko.me`` URL or a bare report code.
        followup:
            When ``True``, uses ``interaction.followup.send`` instead of
            ``interaction.response.send_message``.
        """
        async def _send(content: str, *, ephemeral: bool = False) -> None:
            if followup:
                await interaction.followup.send(content, ephemeral=ephemeral)
            else:
                await interaction.response.send_message(content, ephemeral=ephemeral)

        # Fetch report -------------------------------------------------------
        try:
            report: SparkReport = await fetch_report(url_or_code, session=self.session)
        except ValueError as exc:
            await _send(f'❌ {exc}', ephemeral=True)
            return
        except RuntimeError as exc:
            await _send(
                f'❌ Falha ao carregar o relatório Spark: {exc}', ephemeral=True
            )
            return
        except Exception:
            logger.exception("Unexpected error fetching Spark report for %r", url_or_code)
            await _send(
                '❌ Erro inesperado ao buscar o relatório. Tente novamente.',
                ephemeral=True,
            )
            return

        # Delegate to DocsRAG ------------------------------------------------
        docs_rag = self.bot.get_cog('DocsRAG')
        if docs_rag is None:
            await _send(
                '⚠️ O módulo de análise (DocsRAG) não está disponível.',
                ephemeral=True,
            )
            return

        await docs_rag.run_spark_analysis(interaction, report)  # type: ignore[attr-defined]

    # --- Passive URL detection ---

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Detect Spark URLs in chat and offer an analysis button."""
        if message.author.bot:
            return
        match = SPARK_URL_PATTERN.search(message.content)
        if not match:
            return
        code = match.group(1)
        view = SparkAnalyzeView(self, code)
        reply = await message.reply(
            '🔥 Relatório Spark detectado! Deseja analisar com IA?',
            view=view,
            mention_author=False,
        )
        view.message = reply


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(SparkAnalyzer(bot))
