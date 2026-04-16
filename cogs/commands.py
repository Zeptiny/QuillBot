import discord
from discord import app_commands
from discord.ext import commands


class Commands(commands.Cog):
    """Slash commands for server administration help."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="plov", description="Informações necessárias para escolher um serviço de hospedagem")
    async def plov(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="PLOV - Informações para Hospedagem",
            color=discord.Color.blue(),
        )
        embed.add_field(name="P - Plano/Players", value="Quais recursos procura", inline=False)
        embed.add_field(name="L - Localização", value="Onde", inline=False)
        embed.add_field(name="O - Orçamento", value="Quanto está disposto a pagar", inline=False)
        embed.add_field(name="V - Versão", value="Qual a versão do servidor", inline=False)
        embed.set_footer(text="Se está em dúvida em qual plano precise, use /plano")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="plano", description="Informações para recomendar um plano adequado")
    async def plano(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Informações para Recomendação de Plano",
            color=discord.Color.green(),
        )
        embed.add_field(name="Versão", value="Qual a versão do servidor", inline=False)
        embed.add_field(name="Players", value="Quantidade de players simultâneos", inline=False)
        embed.add_field(name="Mods/Plugins", value="Quantos mods/plugins (Especifique)", inline=False)
        embed.add_field(name="Modo", value="Qual o modo de jogo do server", inline=False)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="docs", description="Link para a documentação do Miners' Refuge")
    @app_commands.describe(busca="Termo para pesquisar na documentação (opcional)")
    async def docs(self, interaction: discord.Interaction, busca: str | None = None):
        base_url = "https://docs.minersrefuge.com.br"
        embed = discord.Embed(
            title="Documentação - Miners' Refuge",
            color=discord.Color.gold(),
        )
        if busca:
            embed.description = (
                f"Pesquise por **{busca}** na documentação:\n"
                f"[Abrir documentação]({base_url})\n\n"
                f"Use `CTRL+K` no site para pesquisar diretamente!"
            )
        else:
            embed.description = (
                f"Acesse a documentação completa:\n"
                f"[docs.minersrefuge.com.br]({base_url})\n\n"
                f"Encontre guias sobre administração de servidores Minecraft, "
                f"dicas de otimização e muito mais!"
            )
        embed.set_footer(text="Contribua abrindo um PR no GitHub!")
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
