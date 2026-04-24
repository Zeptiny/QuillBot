import logging

import discord
from discord import app_commands
from discord.ext import commands

from config import BOT_TOKEN, validate_config

validate_config()
assert BOT_TOKEN is not None  # narrowed by validate_config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('quillbot')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Load order matters: log_analyzer's on_message (pattern matching) runs before
# docs_rag's on_message (follow-up replies). Do not reorder without reviewing
# listener interactions.
COGS = ['cogs.log_analyzer', 'cogs.commands', 'cogs.plugins', 'cogs.spark', 'cogs.docs_rag']


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f'⏳ Aguarde {error.retry_after:.0f}s antes de usar este comando novamente.',
            ephemeral=True,
        )
        return
    logger.exception("Unhandled command error: %s", error)
    msg = 'Ocorreu um erro inesperado. Tente novamente.'
    if interaction.response.is_done():
        await interaction.followup.send(msg, ephemeral=True)
    else:
        await interaction.response.send_message(msg, ephemeral=True)


@bot.event
async def on_ready():
    logger.info('Logged on as %s', bot.user)
    synced = await bot.tree.sync()
    logger.info('Synced %d slash command(s) with Discord', len(synced))


async def main():
    async with bot:
        for cog in COGS:
            await bot.load_extension(cog)
            logger.info('Loaded cog: %s', cog)
        await bot.start(BOT_TOKEN)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
