"""extract_pingable_mentions: which @names and <@ids> in a reply become real pings."""
from types import SimpleNamespace as NS

import pytest

from cogs.utils import extract_pingable_mentions


class FakeGuild:
    def __init__(self, members):
        self.members = members
        self._by_id = {m.id: m for m in members}

    def get_member(self, uid):
        return self._by_id.get(uid)

    def get_member_named(self, name):
        for m in self.members:
            if name in (m.display_name, m.name):
                return m
        return None


nyuu = NS(id=111111111111111111, bot=False, name='nyuu', display_name='Nyuu', discriminator='1234')
john = NS(id=222222222222222222, bot=False, name='john.doe', display_name='John Doe', discriminator='5678')
joao = NS(id=333333333333333333, bot=False, name='joao', display_name='João', discriminator='9012')
other_bot = NS(id=444444444444444444, bot=True, name='otherbot', display_name='OtherBot', discriminator='0001')
g = FakeGuild([nyuu, john, joao, other_bot])

CASES = [
    ('Olá <@111111111111111111>! Hora de dormir.', g, '<@111111111111111111>'),
    ('Ei @nyuu, vai dormir!', g, '<@111111111111111111>'),
    ('Ei @John Doe, vai dormir!', g, '<@222222222222222222>'),
    ('Ei @Joao (sem acento)', g, '<@333333333333333333>'),
    ('@everyone @here vão dormir @nyuu', g, '<@111111111111111111>'),
    ('<@&999999999999999999> <@111111111111111111>', g, '<@111111111111111111>'),
    ('<@444444444444444444> @OtherBot acorda', g, ''),
    ('contato: john.doe@example.com e @nyuu', g, '<@111111111111111111>'),
    ('@fantasma <@123456789012345678>', g, ''),
    ('<@111111111111111111> @nyuu <@!111111111111111111>', g, '<@111111111111111111>'),
    ('@nyuu', None, ''),
]


@pytest.mark.parametrize('text, guild, expected', CASES)
def test_extract_pingable_mentions(text, guild, expected):
    assert extract_pingable_mentions(text, guild) == expected


def test_limit():
    r = extract_pingable_mentions('@nyuu @joao @John Doe', g, limit=2)
    assert r == '<@111111111111111111> <@333333333333333333>'
