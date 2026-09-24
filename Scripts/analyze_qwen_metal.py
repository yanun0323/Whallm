#!/usr/bin/env python3
"""Join synchronized Qwen scope timestamps to target-PID Metal active intervals.

This measures temporal activity, not kernel ownership, occupancy or bandwidth.
Export time-info and metal-gpu-intervals tables with xctrace before running this.
"""
import argparse
from bisect import bisect_left, bisect_right
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def table_rows(path):
    """Resolve native xctrace XML's per-export backward ID/ref cells."""
    refs, columns = {}, []
    for _, node in ET.iterparse(path, events=('end',)):
        if node.get('id'):
            refs[node.get('id')] = (node.text, node.get('fmt', ''))
        if node.tag == 'schema':
            columns = [col.findtext('mnemonic') for col in node.findall('col')]
        if node.tag == 'row':
            if len(node) != len(columns):
                raise ValueError('incomplete xctrace row')
            result = {}
            for column, cell in zip(columns, node):
                value, display = refs[cell.get('ref')] if cell.get('ref') else (cell.text, cell.get('fmt', ''))
                result[column] = value if value is not None else display
            yield result
            node.clear()


def merge_intervals(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if hi < lo:
            raise ValueError('negative GPU interval')
        if hi == lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def summarize(timeline, gpu_rows, time_info, *, pid):
    metadata, *events = timeline
    if metadata.get('clock') != 'mach_absolute_time' or metadata.get('mode') != 'sync' or metadata.get('pid') != pid:
        raise ValueError('GPU scope attribution requires matching PID and synchronized Mach timeline')
    epoch = int(time_info['mabs-epoch'])
    numerator, denominator = map(int, time_info['timebase-info'].split('/'))
    if numerator <= 0 or denominator <= 0:
        raise ValueError('invalid trace timebase')
    active = []
    for row in gpu_rows:
        if row['state'] == 'Active' and str(row['process']).endswith(f'({pid})'):
            start = int(row['start'])
            active.append((start, start + int(row['duration']), row['cmdbuffer-id']))
    if not active:
        raise ValueError('no target GPU evidence; missing counters are not zero utilization')
    union = merge_intervals((a,b) for a,b,_ in active)
    starts, ends = [x[0] for x in union], [x[1] for x in union]
    windows = []
    for event in events:
        if event['failed']:
            raise ValueError('incomplete profiling scope')
        lo = (event['begin']-epoch)*numerator/denominator
        hi = (event['end']-epoch)*numerator/denominator
        if lo < 0 or hi <= lo:
            raise ValueError('invalid timeline window')
        windows.append((lo,hi,event))
    decoders = sorted((a,b) for a,b,e in windows if e['kind']=='decoder')
    if any(a[1]>b[0] for a,b in zip(decoders,decoders[1:])):
        raise ValueError('overlapping decoder scopes cannot be added')
    phases = [(a,b,e) for a,b,e in windows if e['kind']=='phase']
    if not phases or max(b for a,b,_ in active) < max(b for a,b,_ in phases)-100_000_000:
        raise ValueError('trace does not cover the request tail')
    grouped = {}
    for lo,hi,event in windows:
        if event['kind'] == 'decoder' and not any(a<=lo and hi<=b and
                (p['request'],p['phase'])==(event['request'],event['phase']) for a,b,p in phases):
            raise ValueError('decoder outside matching request phase')
        # Merged ends/starts are monotonic, so only overlapping intervals are visited.
        relevant = union[bisect_right(ends,lo):bisect_left(starts,hi)]
        busy = sum(min(b,hi)-max(a,lo) for a,b in relevant)
        key = (event['kind'],event['request'],event['phase'],event['layer'])
        entry = grouped.setdefault(key, dict(kind=key[0],request=key[1],phase=key[2],layer=key[3],
            calls=0,wall_seconds=0.,gpu_active_seconds=0.,buffers=set()))
        entry['calls'] += 1
        entry['wall_seconds'] += (hi-lo)/1e9
        entry['gpu_active_seconds'] += busy/1e9
        entry['buffers'].update(c for a,b,c in active if a<hi and b>lo)
    rows = []
    for row in grouped.values():
        row['command_buffer_ids'] = len(row.pop('buffers'))
        row['gpu_active_percent'] = 100*row['gpu_active_seconds']/row['wall_seconds']
        row['no_target_gpu_activity_seconds'] = row['wall_seconds']-row['gpu_active_seconds']
        rows.append(row)
    return sorted(rows,key=lambda r:(r['request'],r['phase'],r['kind'],r['layer']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--timeline',type=Path,required=True)
    parser.add_argument('--gpu-table',type=Path,required=True)
    parser.add_argument('--time-info',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if result['status'] != 'completed' or result['mode'] != 'sync':
        parser.error('requires a completed sync result')
    times = list(table_rows(args.time_info))
    if len(times)!=1:
        parser.error('requires one uninterrupted trace timebase')
    rows = summarize([json.loads(line) for line in args.timeline.read_text().splitlines()],
        table_rows(args.gpu_table),times[0],pid=result['pid'])
    actual = {(r['request'],r['phase'],r['layer']):r['calls'] for r in rows if r['kind']=='decoder'}
    expected = {(r['request'],r['phase'],r['layer']):r['calls'] for r in result['profile']['rows'] if r['component']=='decoder'}
    if not expected or actual != expected:
        parser.error('timeline does not match nonempty profiled decoder calls')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'gpu-scopes.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'gpu-scopes.json').write_text(json.dumps(dict(pid=result['pid'],rows=rows,
        limits='Target-PID Active encoder interval union overlapping synchronized scopes. Not kernel ownership, hardware occupancy, memory bandwidth, normal throughput or reclaimable idle time. Lazy dependencies can execute inside a scope. Nested phase/decoder rows must not be summed; buffer IDs can cross boundaries.'),indent=2)+'\n')
    print(f'Validated {len(actual)} decoder aggregates; {len(rows)} GPU scope rows')


if __name__=='__main__':main()
