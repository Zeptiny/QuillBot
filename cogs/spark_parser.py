"""Spark profiler report parser — shared module used by the Spark cog and DocsRAG.

Architecture
------------
SparkReport      — dataclass holding every extracted field from the JSON report.
                   Never fully serialised to the LLM.

build_summary()  — compact (~1.7 k chars) text injected as the LLM's initial
                   context. Covers identity, health signals, hotspots and key
                   config values.

build_detail(r, section)
                 — on-demand section dispatcher called by LLM tool calls:
                     "hotspots"               full call tree (chain-collapsed)
                     "jvm"                    JVM flags + GC detail
                     "profiler"               time-window TPS/MSPT trend table
                     "plugins"                full plugin list with versions
                     "world"                  per-world entity breakdown
                     "game_rules"             non-default gamerules first
                     "configs:<filename>"     single config file, e.g.
                                              "configs:server.properties"
                                              "configs:paper/"

fetch_report(url_or_code, session=None)
                 — async fetch from spark-json-service. Accepts an existing
                   aiohttp.ClientSession or creates a disposable one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    pass  # forward references only

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARK_URL_PATTERN = re.compile(r'https://spark\.lucko\.me/(\w+)')
SPARK_JSON_SERVICE = 'https://spark-json-service.lucko.me'

# Maximum response body before JSON parsing (prevents memory exhaustion)
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB

_SAMPLER_MODES = {0: 'EXECUTION', 1: 'ALLOCATION'}
_SAMPLER_ENGINES = {0: 'JAVA', 1: 'ASYNC'}

_SUMMARY_PROPS = (
    'view-distance',
    'simulation-distance',
    'difficulty',
    'gamemode',
    'online-mode',
    'pvp',
    'max-players',
    'spawn-monsters',
    'spawn-protection',
    'network-compression-threshold',
    'max-tick-time',
    'sync-chunk-writes',
)

# Enum-like list exposed to the LLM via the get_spark_detail tool description.
AVAILABLE_SECTIONS = (
    'hotspots',
    'jvm',
    'profiler',
    'plugins',
    'world',
    'game_rules',
    'configs:server.properties',
    'configs:bukkit.yml',
    'configs:spigot.yml',
    'configs:paper/',
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SparkReport:
    # --- report type ---
    report_type: str = ''
    spark_code: str = ''

    # --- server identity ---
    platform_name: str = ''
    platform_brand: str = ''
    platform_version: str = ''
    minecraft_version: str = ''

    # --- java / JVM ---
    java_version: str = ''
    java_vendor: str = ''
    java_vendor_version: str = ''
    jvm_name: str = ''
    jvm_flags: str = ''
    xmx_mb: int = 0
    xms_mb: int = 0

    # --- system ---
    os_name: str = ''
    os_arch: str = ''
    os_version: str = ''
    cpu_model: str = ''
    cpu_threads: int = 0
    sys_ram_used_gb: float = 0.0
    sys_ram_total_gb: float = 0.0
    sys_swap_used_gb: float = 0.0
    sys_swap_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0

    # --- uptime ---
    uptime_s: float = 0.0

    # --- performance snapshot ---
    tps_1m: float = 0.0
    tps_5m: float = 0.0
    tps_15m: float = 0.0
    tps_target: int = 20
    mspt_median: float = 0.0
    mspt_max: float = 0.0
    mspt_p95: float = 0.0
    mspt_target: float = 50.0
    heap_used_mb: float = 0.0
    heap_max_mb: float = 0.0
    player_count: int = 0
    ping_median_ms: float = 0.0

    # --- GC ---
    platform_gc: dict = field(default_factory=dict)
    system_gc: dict = field(default_factory=dict)

    # --- profiler runtime ---
    sampler_mode: str = ''
    sampler_engine: str = ''
    interval_us: int = 0
    start_time_ms: int = 0
    end_time_ms: int = 0
    number_of_ticks: int = 0
    comment: str = ''

    # --- time-window trend ---
    time_windows: list[dict] = field(default_factory=list)

    # --- world ---
    total_entities: int = 0
    entity_counts: dict = field(default_factory=dict)
    worlds: list[dict] = field(default_factory=list)
    game_rules: list[dict] = field(default_factory=list)
    data_packs: list[dict] = field(default_factory=list)

    # --- plugins / mods ---
    plugins: list[dict] = field(default_factory=list)

    # --- server configurations (full, parsed per file) ---
    configs: dict = field(default_factory=dict)

    # --- profiler filter / thread scope ---
    tick_length_threshold_ms: int = 0   # >0 = --only-ticks-over (lag spike profile)
    ticks_included: int = 0             # how many ticks passed the filter
    data_aggregator_type: str = ''      # SIMPLE | TICKED
    thread_grouper: str = ''            # AS_ONE | BY_NAME | BY_POOL
    profiled_thread_ids: list[int] = field(default_factory=list)

    # --- plugin source map (class name → plugin name) ---
    class_sources: dict = field(default_factory=dict)

    # --- hotspots ---
    hotspots: list[dict] = field(default_factory=list)
    hotspot_tree: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_report(data: dict, code: str = '') -> SparkReport:
    r = SparkReport()
    r.spark_code = code
    r.report_type = data.get('type', 'sampler')

    meta = data.get('metadata') or {}

    # platform
    platform = meta.get('platform') or {}
    r.platform_name = platform.get('name', '')
    r.platform_brand = platform.get('brand', r.platform_name)
    r.platform_version = platform.get('version', '')
    r.minecraft_version = platform.get('minecraftVersion', '')

    # system stats
    sys_stats = meta.get('systemStatistics') or {}
    java = sys_stats.get('java') or {}
    r.java_version = java.get('version', '')
    r.java_vendor = java.get('vendor', '')
    r.java_vendor_version = java.get('vendorVersion', '')
    r.jvm_flags = java.get('vmArgs', '')
    r.xmx_mb, r.xms_mb = _parse_heap_flags(r.jvm_flags)

    jvm = sys_stats.get('jvm') or {}
    r.jvm_name = jvm.get('name', '')

    os_ = sys_stats.get('os') or {}
    r.os_name = os_.get('name', '')
    r.os_arch = os_.get('arch', '')
    r.os_version = os_.get('version', '')

    cpu = sys_stats.get('cpu') or {}
    r.cpu_model = cpu.get('modelName', '')
    r.cpu_threads = cpu.get('threads', 0)

    sys_mem = sys_stats.get('memory') or {}
    phys = sys_mem.get('physical') or {}
    r.sys_ram_used_gb = phys.get('used', 0) / 1_073_741_824
    r.sys_ram_total_gb = phys.get('total', 0) / 1_073_741_824
    swap = sys_mem.get('swap') or {}
    r.sys_swap_used_gb = swap.get('used', 0) / 1_073_741_824
    r.sys_swap_total_gb = swap.get('total', 0) / 1_073_741_824

    disk = sys_stats.get('disk') or {}
    r.disk_used_gb = disk.get('used', 0) / 1_073_741_824
    r.disk_total_gb = disk.get('total', 0) / 1_073_741_824

    r.system_gc = sys_stats.get('gc') or {}

    # platform stats
    plat_stats = meta.get('platformStatistics') or {}
    r.uptime_s = (plat_stats.get('uptime') or 0) / 1000

    tps = plat_stats.get('tps') or {}
    r.tps_1m = tps.get('last1m', 0.0)
    r.tps_5m = tps.get('last5m', 0.0)
    r.tps_15m = tps.get('last15m', 0.0)
    r.tps_target = tps.get('gameTargetTps') or 20

    mspt = plat_stats.get('mspt') or {}
    mspt_1m = mspt.get('last1m') or {}
    r.mspt_median = mspt_1m.get('median', 0.0)
    r.mspt_max = mspt_1m.get('max', 0.0)
    r.mspt_p95 = mspt_1m.get('percentile95', 0.0)
    r.mspt_target = (mspt.get('gameMaxIdealMspt') or 0) or (1000 / r.tps_target)

    heap = (plat_stats.get('memory') or {}).get('heap') or {}
    r.heap_used_mb = heap.get('used', 0) / 1_048_576
    r.heap_max_mb = heap.get('max', 0) / 1_048_576

    r.player_count = plat_stats.get('playerCount') or 0
    ping_stat = (plat_stats.get('ping') or {}).get('last15m') or {}
    r.ping_median_ms = ping_stat.get('median', 0.0)

    r.platform_gc = plat_stats.get('gc') or {}

    # world
    world = plat_stats.get('world') or {}
    r.total_entities = world.get('totalEntities', 0)
    r.entity_counts = world.get('entityCounts') or {}
    r.worlds = world.get('worlds') or []
    r.game_rules = world.get('gameRules') or []
    r.data_packs = world.get('dataPacks') or []

    # profiler metadata
    r.sampler_mode = _SAMPLER_MODES.get(meta.get('samplerMode', 0), 'EXECUTION')
    r.sampler_engine = _SAMPLER_ENGINES.get(meta.get('samplerEngine', 0), 'JAVA')
    r.interval_us = meta.get('interval', 0)
    r.start_time_ms = meta.get('startTime', 0)
    r.end_time_ms = meta.get('endTime', 0)
    r.number_of_ticks = meta.get('numberOfTicks', 0)
    r.comment = meta.get('comment', '')

    # time-window trend
    tws = data.get('timeWindowStatistics') or {}
    r.time_windows = sorted(tws.values(), key=lambda w: w.get('startTime', 0))

    # plugins (sources map)
    sources = meta.get('sources') or {}
    r.plugins = [
        {
            'name': v.get('name', k),
            'version': v.get('version', ''),
            'author': v.get('author', ''),
            'description': v.get('description', ''),
            'builtin': v.get('builtIn', False),
        }
        for k, v in sources.items()
    ]

    # profiler filter: dataAggregator reveals --only-ticks-over and thread grouping
    _AGGREGATOR_TYPES = {0: 'SIMPLE', 1: 'TICKED'}
    _THREAD_GROUPERS = {0: 'AS_ONE', 1: 'BY_NAME', 2: 'BY_POOL'}
    da = meta.get('dataAggregator') or {}
    r.data_aggregator_type = _AGGREGATOR_TYPES.get(da.get('type', 0), 'SIMPLE')
    r.thread_grouper = _THREAD_GROUPERS.get(da.get('threadGrouper', 1), 'BY_NAME')
    r.tick_length_threshold_ms = da.get('tickLengthThreshold', 0)
    r.ticks_included = da.get('numberOfIncludedTicks', 0)

    # threadDumper: which threads were actually profiled
    td = meta.get('threadDumper') or {}
    r.profiled_thread_ids = td.get('ids') or []

    # classSources: maps class name → source index → plugin name
    src_idx_to_name: dict[int, str] = {}
    for k, v in sources.items():
        try:
            src_idx_to_name[int(k)] = v.get('name', k)
        except (ValueError, TypeError):
            pass
    r.class_sources = {
        cls: src_idx_to_name.get(idx, str(idx))
        for cls, idx in (data.get('classSources') or {}).items()
    }

    # configs — JSON-encoded values are decoded; plain dicts/strings kept as-is
    raw_configs = meta.get('serverConfigurations') or {}
    for cfg_name, cfg_value in raw_configs.items():
        if isinstance(cfg_value, str):
            try:
                r.configs[cfg_name] = json.loads(cfg_value)
            except (json.JSONDecodeError, TypeError):
                r.configs[cfg_name] = cfg_value
        else:
            r.configs[cfg_name] = cfg_value or {}

    # hotspots
    threads = data.get('threads') or []
    r.hotspots = _extract_hotspots(threads)
    r.hotspot_tree = _extract_hotspot_tree(threads, r.class_sources)

    return r


def _parse_heap_flags(vm_args: str) -> tuple[int, int]:
    def _to_mb(m: re.Match) -> int:
        val, unit = int(m.group(1)), m.group(2).upper()
        return val * 1024 if unit == 'G' else val

    xmx = re.search(r'-Xmx(\d+)([MmGg])', vm_args)
    xms = re.search(r'-Xms(\d+)([MmGg])', vm_args)
    return (_to_mb(xmx) if xmx else 0), (_to_mb(xms) if xms else 0)


def _extract_hotspots(threads: list[dict]) -> list[dict]:
    result = []
    for thread in threads:
        thread_total = sum(thread.get('times', [])) or thread.get('time', 0)
        if thread_total <= 0:
            continue
        nodes: list[tuple[float, str]] = []
        _walk_nodes(thread.get('children', []), thread_total, nodes)
        nodes.sort(reverse=True)
        seen: set[str] = set()
        unique = []
        for pct, label in nodes:
            if label not in seen:
                seen.add(label)
                unique.append((pct, label))
        result.append({'thread': thread.get('name', '?'), 'nodes': unique})
    return result


def _walk_nodes(
    nodes: list[dict],
    thread_total: float,
    acc: list[tuple[float, str]],
    depth: int = 0,
    max_depth: int = 12,
) -> None:
    if depth > max_depth:
        return
    for node in nodes:
        node_time = sum(node.get('times', [])) or node.get('time', 0)
        if node_time <= 0:
            continue
        cn = node.get('className', '')
        mn = node.get('methodName', '')
        if cn:
            acc.append((node_time / thread_total * 100, f'{cn}.{mn}()' if mn else cn))
        _walk_nodes(node.get('children', []), thread_total, acc, depth + 1, max_depth)


# ---------------------------------------------------------------------------
# Hierarchical call-tree extraction (for _detail_hotspots and smart summary)
# ---------------------------------------------------------------------------

def _extract_hotspot_tree(
    threads: list[dict],
    class_sources: dict | None = None,
) -> list[dict]:
    """Build per-thread pruned call trees preserving hierarchy.

    Spark stores per-thread frames as a flat array where each node carries a
    ``childrenRefs`` list of indices (into that same array) pointing to the
    frames it called.  The root frame(s) are those whose index does not appear
    in any other node's ``childrenRefs``.
    """
    class_sources = class_sources or {}
    result = []
    for thread in threads:
        thread_total = sum(thread.get('times', [])) or thread.get('time', 0)
        if thread_total <= 0:
            continue
        flat_nodes = thread.get('children', [])
        if not flat_nodes:
            continue
        # Identify root indices: not referenced by any other node.
        referenced: set[int] = set()
        for node in flat_nodes:
            for ref in (node.get('childrenRefs') or []):
                referenced.add(ref)
        roots = [i for i in range(len(flat_nodes)) if i not in referenced]
        tree_children = [
            n for idx in roots
            if (n := _build_pruned_tree_ref(
                flat_nodes, idx, thread_total, class_sources=class_sources
            )) is not None
        ]
        result.append({
            'thread': thread.get('name', '?'),
            'total': thread_total,
            'children': tree_children,
        })
    return result


def _build_pruned_tree_ref(
    flat_nodes: list[dict],
    idx: int,
    thread_total: float,
    depth: int = 0,
    max_depth: int = 20,
    class_sources: dict | None = None,
) -> dict | None:
    """Recursively build a pruned tree node using index-based childrenRefs lookup.

    Each returned node carries:
      pct      – % of thread total samples (total time, incl. sub-calls)
      self_pct – % spent *inside* this method (not in sub-calls); the real cost
      source   – plugin/mod name when classSources data is present, else None
    """
    if class_sources is None:
        class_sources = {}
    if idx >= len(flat_nodes):
        return None
    node = flat_nodes[idx]
    node_time = sum(node.get('times', [])) or node.get('time', 0)
    if not node_time or thread_total <= 0:
        return None
    pct = node_time / thread_total * 100
    if pct < 0.5:
        return None
    cn = node.get('className', node.get('name', ''))
    mn = node.get('methodName', '')
    label = f'{cn}.{mn}()' if mn else cn
    source: str | None = class_sources.get(cn)
    children: list[dict] = []
    if depth < max_depth:
        for child_ref in (node.get('childrenRefs') or []):
            child_node = _build_pruned_tree_ref(
                flat_nodes, child_ref, thread_total, depth + 1, max_depth,
                class_sources=class_sources,
            )
            if child_node:
                children.append(child_node)
    # Self time = node's own time minus what sub-calls account for.
    child_time_sum = sum(c['_time'] for c in children)
    self_time = max(0, node_time - child_time_sum)
    self_pct = self_time / thread_total * 100
    return {
        'label': label,
        'pct': pct,
        'self_pct': self_pct,
        '_time': node_time,  # internal; used by parent for self-time calculation
        'source': source,
        'children': children,
    }


def _render_tree(
    node: dict,
    prefix: str = '',
    is_last: bool = True,
    lines: list[str] | None = None,
) -> list[str]:
    """Render a call-tree node with box-drawing chars, collapsing linear chains."""
    if lines is None:
        lines = []
    connector = '\u2514\u2500 ' if is_last else '\u251c\u2500 '
    source_tag = f'  [{node["source"]}]' if node.get('source') else ''
    self_tag = f'  (self {node["self_pct"]:.1f}%)' if node.get('self_pct', 0) >= 1.0 else ''
    lines.append(f'{prefix}{connector}{node["pct"]:5.1f}%  {node["label"]}{source_tag}{self_tag}')
    child_prefix = prefix + ('   ' if is_last else '\u2502  ')
    children = node['children']
    if not children:
        return lines
    if len(children) == 1:
        # Walk the single-child chain as long as the child carries ≥90 % of
        # the current node's time (pure pass-through / dispatcher frames).
        skipped = 0
        current = children[0]
        while (
            len(current['children']) == 1
            and current['children'][0]['pct'] >= current['pct'] * 0.90
        ):
            skipped += 1
            current = current['children'][0]
        if skipped >= 2:
            frame_word = 'frames' if skipped > 1 else 'frame'
            lines.append(
                f'{child_prefix}   \u00b7\u00b7\u00b7 {skipped} intermediate '
                f'{frame_word} collapsed \u00b7\u00b7\u00b7'
            )
        _render_tree(current, child_prefix, is_last=True, lines=lines)
    else:
        for i, child in enumerate(children):
            _render_tree(child, child_prefix, is_last=(i == len(children) - 1), lines=lines)
    return lines


def _fmt_hotspots_smart(hotspot_tree: list[dict]) -> list[str]:
    """Compact per-thread summary using chain-collapsed tree rendering.

    Single-child pass-through chains are collapsed (≥2 consecutive frames where
    the child carries ≥90 % of the parent's time).  This means:
    - An idle server sleeping in parkNanos/libc collapses to ~3 lines.
    - A lagging server shows each branch of runServer() explicitly.
    """
    lines: list[str] = []
    for ht in hotspot_tree:
        lines.append(f'{ht["thread"]}:')
        if not ht['children']:
            lines.append('  (no significant samples)')
            continue
        for i, child in enumerate(ht['children']):
            _render_tree(child, prefix='  ', is_last=(i == len(ht['children']) - 1),
                         lines=lines)
    return lines


# ---------------------------------------------------------------------------
# Summary — sent to LLM as initial context (compact)
# ---------------------------------------------------------------------------

def build_summary(r: SparkReport) -> str:
    lines: list[str] = [
        '=== Spark Profiler Report ===',
    ]
    if r.spark_code:
        lines.append(f'URL       : https://spark.lucko.me/{r.spark_code}')

    # lag spike profile banner (most important context for interpretation)
    if r.tick_length_threshold_ms > 0:
        incl = f', {r.ticks_included} ticks included' if r.ticks_included else ''
        lines += [
            '',
            f'\u26a0 LAG SPIKE PROFILE — only ticks >{r.tick_length_threshold_ms}ms were sampled{incl}.',
            '  Hotspots below represent the CAUSE of lag spikes, not normal server load.',
        ]

    # profiler runtime
    duration_s = (r.end_time_ms - r.start_time_ms) / 1000 if r.end_time_ms else 0
    start_dt = _fmt_ts(r.start_time_ms)
    lines += [
        '',
        '--- Profiler Runtime ---',
        f'Mode      : {r.sampler_mode} ({r.sampler_engine} engine, '
        f'interval {r.interval_us / 1000:.1f}ms)',
        f'Duration  : {duration_s:.0f}s  |  {r.number_of_ticks} ticks  |  started {start_dt}',
    ]
    if r.comment:
        lines.append(f'Comment   : {r.comment}')
    if r.thread_grouper and r.thread_grouper != 'BY_NAME':
        lines.append(f'Grouper   : {r.thread_grouper}')

    # server identity
    lines += [
        '',
        '--- Server ---',
        f'Software  : {r.platform_brand} {r.platform_version}',
        f'Minecraft : {r.minecraft_version}',
        f'Java      : {r.java_version} ({r.java_vendor})',
        f'JVM heap  : -Xms{r.xms_mb}M  -Xmx{r.xmx_mb}M',
        f'OS        : {r.os_name} ({r.os_arch})',
        f'Uptime    : {_fmt_uptime(r.uptime_s)}  |  Players: {r.player_count}',
    ]

    # performance
    tps_tag = _tps_health(r.tps_1m, r.tps_target)
    mspt_tag = _mspt_health(r.mspt_median, r.mspt_target)
    heap_pct = r.heap_used_mb / r.heap_max_mb * 100 if r.heap_max_mb else 0
    lines += [
        '',
        '--- Performance ---',
        f'TPS       : {r.tps_1m:.2f} (1m) / {r.tps_5m:.2f} (5m) / '
        f'{r.tps_15m:.2f} (15m)  [target {r.tps_target}]  {tps_tag}',
        f'MSPT      : median {r.mspt_median:.2f}ms  max {r.mspt_max:.2f}ms  '
        f'p95 {r.mspt_p95:.2f}ms  [ideal \u2264{r.mspt_target:.0f}ms]  {mspt_tag}',
        f'Heap      : {r.heap_used_mb:.0f}MB / {r.heap_max_mb:.0f}MB ({heap_pct:.0f}% used)',
    ]
    if r.ping_median_ms:
        lines.append(f'Ping      : {r.ping_median_ms:.1f}ms avg (15m)')

    # GC (system-level, JVM lifetime — more representative than platform GC)
    gc_lines = _fmt_gc(r.system_gc)
    if gc_lines:
        lines += ['', '--- GC (JVM lifetime) ---'] + gc_lines
    else:
        lines.append('GC        : no collections recorded')

    # system
    lines += [
        '',
        '--- System ---',
        f'CPU       : {r.cpu_model} ({r.cpu_threads} threads)',
        f'RAM       : {r.sys_ram_used_gb:.1f}GB / {r.sys_ram_total_gb:.1f}GB',
        f'Disk      : {r.disk_used_gb:.1f}GB / {r.disk_total_gb:.1f}GB',
    ]

    # key server.properties
    sp = r.configs.get('server.properties') or {}
    if isinstance(sp, dict):
        sp_lines = [f'  {k} = {sp[k]}' for k in _SUMMARY_PROPS if k in sp]
        if sp_lines:
            lines += ['', '--- Key Config (server.properties) ---'] + sp_lines

    # world snapshot
    if r.total_entities:
        top_mobs = sorted(r.entity_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        mob_str = ', '.join(f'{n}\u00d7{t}' for t, n in top_mobs)
        lines += ['', '--- World ---', f'Entities  : {r.total_entities} total  ({mob_str})']
        dim_names = ', '.join(w['name'] for w in r.worlds)
        if dim_names:
            lines.append(f'Dimensions: {dim_names}')

    # plugins
    third_party = [p for p in r.plugins if not p['builtin']]
    if third_party:
        plugin_str = ', '.join(
            f"{p['name']} {p['version']}".strip() for p in third_party
        )
        lines += ['', f'--- Plugins ({len(third_party)}) ---', plugin_str]

    # lag spike window detection from per-window statistics
    spike_windows = [
        (i + 1, w)
        for i, w in enumerate(r.time_windows)
        if w.get('msptMax', 0) > r.mspt_target and r.mspt_target > 0
    ]
    if spike_windows:
        lines += ['', '--- Lag Spike Windows ---']
        for win_num, w in spike_windows[:5]:  # cap at 5 to keep summary compact
            tps_str = f"TPS {w.get('tps', 0):.1f}"
            lines.append(
                f'  Window {win_num}: {tps_str},  '
                f"MSPT med {w.get('msptMedian', 0):.1f}ms  "
                f"MAX {w.get('msptMax', 0):.1f}ms  \u2190 spike"
            )
        if len(spike_windows) > 5:
            lines.append(f'  ... and {len(spike_windows) - 5} more spike windows')

    # hotspots — smart branch-point summary
    hotspot_lines = _fmt_hotspots_smart(r.hotspot_tree)
    if hotspot_lines:
        lines += ['', '--- CPU Hotspots (first branch) ---'] + hotspot_lines
        lines.append('(use detail section "hotspots" for the full call tree)')
    else:
        lines += ['', '--- CPU Hotspots ---', 'No sampler data (server may be idle).']

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Detail sections — returned on-demand by LLM tool calls
# ---------------------------------------------------------------------------

def build_detail(r: SparkReport, section: str) -> str:
    """Return detailed data for the requested section.

    Accepts either a bare section name (e.g. "hotspots") or a per-file config
    address (e.g. "configs:server.properties").  Section names are
    case-insensitive; config filenames preserve their original casing.
    """
    # Per-file config: "configs:server.properties", "configs:paper/", etc.
    if section.lower().startswith('configs:'):
        filename = section[len('configs:'):]  # preserve original case
        return _detail_config_file(r, filename)

    section = section.lower().strip()
    dispatch = {
        'configs': _detail_configs,
        'game_rules': _detail_game_rules,
        'world': _detail_world,
        'jvm': _detail_jvm,
        'profiler': _detail_profiler_trend,
        'plugins': _detail_plugins,
        'hotspots': _detail_hotspots,
    }
    if section not in dispatch:
        available = ', '.join(
            [*dispatch.keys(), 'configs:<filename>']
        )
        return f'Unknown section "{section}". Available: {available}'
    return dispatch[section](r)


def _detail_config_file(r: SparkReport, filename: str) -> str:
    """Return a single configuration file's content as formatted JSON/text."""
    cfg = r.configs.get(filename)
    if cfg is None:
        available = ', '.join(r.configs.keys()) or 'none'
        return f'Config file "{filename}" not found. Available: {available}'
    lines = [f'=== Config: {filename} ===']
    lines.append(json.dumps(cfg, indent=2) if isinstance(cfg, dict) else str(cfg))
    return '\n'.join(lines)


def _detail_configs(r: SparkReport) -> str:
    """Return all configuration files concatenated (may be large)."""
    lines = ['=== Full Server Configurations ===']
    for cfg_name, cfg_value in r.configs.items():
        lines.append(f'\n--- {cfg_name} ---')
        lines.append(json.dumps(cfg_value, indent=2) if isinstance(cfg_value, dict)
                     else str(cfg_value))
    return '\n'.join(lines)


def _detail_game_rules(r: SparkReport) -> str:
    lines = ['=== Game Rules ===',
             '(only rules that differ from default in at least one world are shown first)']
    non_default = []
    all_default = []
    for rule in r.game_rules:
        name = rule.get('name', '')
        default = str(rule.get('defaultValue', ''))
        world_vals = rule.get('worldValues') or {}
        diff = {w: v for w, v in world_vals.items() if str(v) != default}
        if diff:
            non_default.append(
                f'  {name}: default={default}  '
                + '  '.join(f'{w}={v}' for w, v in diff.items())
            )
        else:
            all_default.append(f'  {name} = {default}')
    if non_default:
        lines += ['', '--- Non-default ---'] + non_default
    lines += ['', '--- All default ---'] + all_default
    return '\n'.join(lines)


def _detail_world(r: SparkReport) -> str:
    lines = [
        '=== World Details ===',
        f'Total entities: {r.total_entities}',
        '',
        '--- Global entity counts ---',
    ]
    for etype, count in sorted(r.entity_counts.items(), key=lambda x: -x[1]):
        lines.append(f'  {etype}: {count}')

    for world in r.worlds:
        lines += ['', f'--- {world["name"]} ({world.get("totalEntities", 0)} entities) ---']
        world_counts: dict[str, int] = {}
        for region in world.get('regions', []):
            for chunk in region.get('chunks', []):
                for etype, cnt in (chunk.get('entityCounts') or {}).items():
                    world_counts[etype] = world_counts.get(etype, 0) + cnt
        for etype, cnt in sorted(world_counts.items(), key=lambda x: -x[1]):
            lines.append(f'  {etype}: {cnt}')

    if r.data_packs:
        lines += ['', '--- Data Packs ---']
        for dp in r.data_packs:
            tag = ' [built-in]' if dp.get('builtIn') else ''
            lines.append(
                f"  {dp.get('name')} ({dp.get('source')}){tag}"
                f"  \u2014 {dp.get('description', '')}"
            )
    return '\n'.join(lines)


def _detail_jvm(r: SparkReport) -> str:
    lines = [
        '=== JVM Details ===',
        f'JVM       : {r.jvm_name}',
        f'Java      : {r.java_vendor_version}',
        f'OS        : {r.os_name} {r.os_version} ({r.os_arch})',
        '',
        '--- JVM Flags ---',
    ]
    lines += [f'  {f}' for f in r.jvm_flags.split()]
    lines += ['', '--- System GC (JVM lifetime) ---']
    lines += _fmt_gc(r.system_gc) or ['  No collections recorded']
    lines += ['', '--- Platform GC (since server start) ---']
    lines += _fmt_gc(r.platform_gc) or ['  No collections recorded']
    lines += [
        '',
        '--- Memory ---',
        f'Heap used : {r.heap_used_mb:.0f}MB / {r.heap_max_mb:.0f}MB',
        f'RAM       : {r.sys_ram_used_gb:.1f}GB / {r.sys_ram_total_gb:.1f}GB',
        f'Swap      : {r.sys_swap_used_gb:.1f}GB / {r.sys_swap_total_gb:.1f}GB',
        f'Disk      : {r.disk_used_gb:.1f}GB / {r.disk_total_gb:.1f}GB',
    ]
    return '\n'.join(lines)


def _detail_profiler_trend(r: SparkReport) -> str:
    duration_s = (r.end_time_ms - r.start_time_ms) / 1000 if r.end_time_ms else 0
    lines = [
        '=== Profiler Time-Window Statistics ===',
        f'Mode      : {r.sampler_mode}  Engine: {r.sampler_engine}',
        f'Interval  : {r.interval_us / 1000:.1f}ms',
        f'Duration  : {duration_s:.0f}s  ({r.number_of_ticks} ticks)',
        '',
        f'{"Win":>4}  {"TPS":>6}  {"MSPT med":>8}  {"MSPT max":>8}  '
        f'{"CPU%":>5}  {"Players":>7}  {"Entities":>8}  {"TileEnt":>7}  {"Chunks":>6}',
        '-' * 72,
    ]
    for i, w in enumerate(r.time_windows, 1):
        lines.append(
            f'{i:>4}  '
            f'{w.get("tps", 0):>6.2f}  '
            f'{w.get("msptMedian", 0):>8.3f}  '
            f'{w.get("msptMax", 0):>8.3f}  '
            f'{w.get("cpuProcess", 0) * 100:>5.1f}  '
            f'{w.get("players", 0):>7}  '
            f'{w.get("entities", 0):>8}  '
            f'{w.get("tileEntities", 0):>7}  '
            f'{w.get("chunks", 0):>6}'
        )
    return '\n'.join(lines)


def _detail_plugins(r: SparkReport) -> str:
    lines = [f'=== Plugins / Mods ({len(r.plugins)}) ===']
    third = [p for p in r.plugins if not p['builtin']]
    builtin = [p for p in r.plugins if p['builtin']]
    if third:
        lines += ['', '--- Third-party ---']
        for p in third:
            author = f" by {p['author']}" if p.get('author') else ''
            desc = f" \u2014 {p['description']}" if p.get('description') else ''
            lines.append(f"  {p['name']} {p['version']}{author}{desc}")
    if builtin:
        lines += ['', '--- Built-in ---']
        for p in builtin:
            lines.append(f"  {p['name']} {p['version']}")
    return '\n'.join(lines)


def _detail_hotspots(r: SparkReport) -> str:
    """Full per-thread call tree, with single-child pass-through chains collapsed."""
    if not r.hotspot_tree:
        return '=== CPU Call Tree ===\nNo sampler data recorded.'
    lines = [
        '=== CPU Call Tree ===',
        'Each node shows % of thread total samples.',
        'Nodes marked (self X%) indicate time spent within that method itself,',
        '  NOT in sub-calls \u2014 this is the actual bottleneck cost.',
        'Nodes marked [Plugin] are attributed to a specific plugin/mod.',
        'Linear pass-through chains are collapsed (\u00b7\u00b7\u00b7 N frames \u00b7\u00b7\u00b7).',
    ]
    if r.tick_length_threshold_ms > 0:
        lines += [
            '',
            f'\u26a0 LAG SPIKE PROFILE: only ticks >{r.tick_length_threshold_ms}ms sampled.',
            '  These hotspots show the cause of lag, not normal server behavior.',
        ]
    lines.append('')
    for ht in r.hotspot_tree:
        lines.append(f'Thread: {ht["thread"]}  ({ht["total"]} samples)')
        if not ht['children']:
            lines.append('  (no significant sample data)')
        else:
            tree_lines: list[str] = []
            for i, child in enumerate(ht['children']):
                _render_tree(child, prefix='  ', is_last=(i == len(ht['children']) - 1),
                             lines=tree_lines)
            lines.extend(tree_lines)
        lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _fmt_uptime(seconds: float) -> str:
    if seconds <= 0:
        return '?'
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f'{h}h {m}m'


def _fmt_ts(ms: int) -> str:
    if not ms:
        return '?'
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M UTC')
    except (OSError, OverflowError):
        return '?'


def _tps_health(tps: float, target: float) -> str:
    if tps >= target * 0.98:
        return '\u2713 healthy'
    if tps >= target * 0.90:
        return '\u26a0 slightly degraded'
    return '\u2717 lagging'


def _mspt_health(mspt: float, target: float) -> str:
    if target <= 0:
        target = 50.0
    if mspt <= target * 0.5:
        return '\u2713 excellent'
    if mspt <= target:
        return '\u2713 healthy'
    if mspt <= target * 1.5:
        return '\u26a0 elevated'
    return '\u2717 critical'


def _fmt_gc(gc: dict) -> list[str]:
    lines = []
    for name, stats in gc.items():
        total = stats.get('total', 0)
        avg_t = stats.get('avgTime', 0)
        avg_f = stats.get('avgFrequency', 0)
        freq_str = f'every {avg_f / 1000:.0f}s' if avg_f > 0 else 'never'
        if total > 0:
            lines.append(f'  {name}: {total} collections, avg {avg_t:.1f}ms, {freq_str}')
    return lines


# ---------------------------------------------------------------------------
# Network fetch
# ---------------------------------------------------------------------------

async def fetch_report(
    url_or_code: str,
    session: aiohttp.ClientSession | None = None,
) -> SparkReport:
    """Fetch and parse a Spark profiler report.

    Parameters
    ----------
    url_or_code:
        Either a full ``https://spark.lucko.me/<code>`` URL or a bare code.
    session:
        An existing ``aiohttp.ClientSession`` to reuse.  If ``None``, a
        disposable session with a 30-second timeout is created.
    """
    match = SPARK_URL_PATTERN.search(url_or_code)
    code = match.group(1) if match else url_or_code.strip('/')
    fetch_url = f'{SPARK_JSON_SERVICE}/{code}?full=true'

    async def _do_fetch(s: aiohttp.ClientSession) -> SparkReport:
        async with s.get(fetch_url) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f'spark-json-service returned HTTP {resp.status}: {body[:200]}'
                )
            # Guard against excessively large payloads before JSON parsing.
            body_bytes = await resp.read()
            if len(body_bytes) > _MAX_RESPONSE_BYTES:
                raise ValueError(
                    f'Response too large ({len(body_bytes):,} bytes > '
                    f'{_MAX_RESPONSE_BYTES:,} byte limit)'
                )
            data = json.loads(body_bytes)
        return parse_report(data, code=code)

    if session is not None:
        return await _do_fetch(session)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as s:
        return await _do_fetch(s)
