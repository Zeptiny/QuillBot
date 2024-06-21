import discord
import checkMessage as cm
from responses.commands import commmandsList
from discord.ext import commands
from dotenv import load_dotenv
import os
import re

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Add commands from dictionary
def addCommand(command_name, response):
    async def command_func(ctx):
        await ctx.send(response)

    bot.command(name=command_name)(command_func)

for commandName, response in commmandsList.items():
    addCommand(commandName, response)

@bot.event
async def on_ready():
    print(f'Logged on as {bot.user}!')

@bot.event
async def on_message(message):
    if message.author == bot.user:  # Ignore bots
        return

    response = cm.checkMessage(message.content)  # Check the message itself for logs and errors

    # Analyze links sent from the user
    mclogsLink = re.search(r'https://mclo.gs/\w+', message.content)
    pastebinLink = re.search(r'https://pastebin.com/\w+', message.content)
    if mclogsLink: link = mclogsLink.group(0)
    elif pastebinLink: link = f'https://pastebin.com/raw/{pastebinLink.group(0).split('/')[-1]}'

    if 'link' in locals():
        await message.add_reaction('👀')  # React so the user knows it was analyzed
        linkContent = await cm.readFileContent(link)
        if linkContent:
            if not response: response = cm.checkMessage(linkContent)

    if message.attachments:  # If the user sent files
        for attachment in message.attachments:
            if attachment.filename.endswith(('.txt', '.log')):
                fileContent = await cm.readFileContent(attachment.url)
                if fileContent:
                    if not response: response = cm.checkMessage(fileContent)  # If a response wasn't already provided
                    link = await cm.uploadMclogs(fileContent)  # Upload to mclo.gs
                    if link:
                        await message.reply(f'Na próxima vez busque utilizar um serviço para enviar suas logs, como o mclo.gs, fiz o upload para você <3: \n <{link}>')
                    else:
                        await message.reply('Algo deu errado ao tentar fazer o upload para o mclo.gs.')
                    if response:
                        break
    if response:
        await message.reply(response)
    await bot.process_commands(message)  # This ensures commands are processed


bot.run(BOT_TOKEN)
