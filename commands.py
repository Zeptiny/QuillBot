from responses.commands import commandsList
def addCommand(command_name, response):
    async def command_func(ctx):
        await ctx.send(response)

    bot.command(name=command_name)(command_func)