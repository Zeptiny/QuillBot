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

    @app_commands.command(name="help", description="Lista todos os comandos disponíveis do bot")
    async def help_command(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="📋 Comandos do QuillBot",
            description="Todos os comandos disponíveis:",
            color=discord.Color.purple(),
        )
        embed.add_field(
            name="🔧 Utilidades",
            value=(
                "**/plov** — Informações para escolher hospedagem (PLOV)\n"
                "**/plano** — Informações para recomendação de plano\n"
                "**/docs** `[busca]` — Link para a documentação\n"
                "**/help** — Esta mensagem"
            ),
            inline=False,
        )
        embed.add_field(
            name="🤖 Inteligência Artificial",
            value=(
                "**/ask** `pergunta` — Pergunte algo sobre servidores Minecraft (busca na documentação)\n"
                "**/analyze** `log_link` ou `log_file` — Análise de logs com IA"
            ),
            inline=False,
        )
        embed.add_field(
            name="⚙️ Administração",
            value="**/reindex** — Re-indexar a documentação (apenas admins)",
            inline=False,
        )
        embed.add_field(
            name="📝 Detecção Automática",
            value=(
                "O bot também analisa automaticamente logs e erros enviados no chat:\n"
                "• Links do **mclo.gs** e **pastebin.com**\n"
                "• Arquivos **.log** e **.txt** anexados\n"
                "• Mensagens com erros conhecidos de Minecraft"
            ),
            inline=False,
        )
        embed.set_footer(text="Miners' Refuge • docs.minersrefuge.com.br")
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(Commands(bot))
