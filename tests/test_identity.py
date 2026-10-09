"""build_identity_block: the model knows its own name/id and the guild it is in."""
from types import SimpleNamespace as NS

from cogs.utils import build_identity_block

BOT = NS(id=900000000000000001, name='quillbot', display_name='Quill')


def test_no_bot_user_gives_empty_block():
    assert build_identity_block(None, None) == ''


def test_names_the_bot_and_its_messages():
    block = build_identity_block(BOT, None)
    assert block.startswith('<identidade>') and block.endswith('</identidade>')
    assert 'Quill (@quillbot)' in block
    assert 'author_id=900000000000000001' in block
    assert 'SUAS respostas' in block
    assert 'mensagem direta' in block


def test_guild_nickname_and_info_are_used():
    guild = NS(
        id=42, name="Miners' Refuge", description='Comunidade\n  de admins',
        owner_id=7, me=NS(display_name='Quill Nick'),
    )
    block = build_identity_block(BOT, guild)
    assert '"Quill Nick (@quillbot)"' in block
    assert "Servidor atual: Miners' Refuge (id=42)." in block
    assert 'Descrição do servidor: Comunidade de admins' in block
    assert 'Dono do servidor: <@7> (id=7).' in block


def test_block_is_stable_for_prefix_caching():
    guild = NS(id=1, name='G', description=None, owner_id=None, me=None, member_count=10)
    first = build_identity_block(BOT, guild)
    guild.member_count = 11
    assert build_identity_block(BOT, guild) == first
    assert 'Descrição' not in first and 'Dono' not in first
