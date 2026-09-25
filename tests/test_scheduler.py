"""Regression smoke test for cron scheduling lifecycle."""

import os
import tempfile
from unittest.mock import patch

from cogs.scheduler import Scheduler, SchedulerStore, _now, _parse_iso


class FakeGuild:
    def __init__(self):
        self.channel = object()

    def get_channel(self, channel_id):
        return self.channel


class FakeBot:
    def __init__(self, guild):
        self.guild = guild

    def get_guild(self, guild_id):
        return self.guild


async def test_cron_job_fires_and_reschedules():
    with tempfile.TemporaryDirectory(prefix='quillbot_scheduler_test_') as temp_dir:
        store = SchedulerStore(os.path.join(temp_dir, 'scheduler.db'))
        store.ensure()
        job = store.create(
            guild_id=1,
            channel_id=2,
            job_type='cron',
            cron_expr='* * * * *',
            next_fire='2020-01-01T00:00:00-03:00',
            prompt='test',
            created_by=3,
            created_by_name='test',
        )

        scheduler = Scheduler(FakeBot(FakeGuild()))
        scheduler.store = store
        prompted = []

        async def run_prompt(fired_job, guild, channel):
            prompted.append(fired_job['id'])

        async def run_synchronously(func, *args, **kwargs):
            return func(*args, **kwargs)

        scheduler._run_scheduled_prompt = run_prompt
        with patch('cogs.scheduler.asyncio.to_thread', run_synchronously):
            await scheduler._fire(job)

        updated = store.get(1, job['id'])
        next_fire = _parse_iso(updated['next_fire'])
        assert updated['fire_count'] == 1, updated
        assert updated['last_fired_at'] is not None, updated
        assert next_fire is not None and next_fire > _now(), updated
        assert prompted == [job['id']], prompted

