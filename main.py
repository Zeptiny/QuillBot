import logging

import discord
from discord.ext import commands

from config import BOT_TOKEN, validate_config

validate_config()

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
COGS = ['cogs.log_analyzer', 'cogs.commands', 'cogs.plugins', 'cogs.docs_rag']


@bot.event
async def on_ready():
    logger.info('Logged on as %s', bot.user)
    try:
        synced = await bot.tree.sync()
        logger.info('Synced %d slash commands', len(synced))
    except Exception:
        logger.exception('Failed to sync slash commands')


async def main():
    async with bot:
        for cog in COGS:
            await bot.load_extension(cog)
            logger.info('Loaded cog: %s', cog)
        await bot.start(BOT_TOKEN)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
