import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('quillbot')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

COGS = ['cogs.log_analyzer', 'cogs.commands', 'cogs.docs_rag']


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
