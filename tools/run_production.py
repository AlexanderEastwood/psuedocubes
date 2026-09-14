#!/usr/bin/env python3
"""Single-GPU pseudocube production service with durable contiguous coverage."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pseudocube.accounting import digest,durable
from pseudocube.focus import Config,Tile,A619,B619
from pseudocube.gpu import Runtime,GPUFocus,Matcher
from pseudocube.production import Frontier,load_production
from review.verify_cube import verify

STOP=False


def stopping(signum: int, frame: Any) -> None:
    global STOP
    STOP=True


def notify(message: str) -> None:
    address=os.environ.get('NOTIFY_SOCKET')
    if address:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as stream:
            stream.connect('\0'+address[1:] if address.startswith('@') else address)
            stream.sendall(message.encode())


def frozen_build() -> str:
    manifest=json.loads((ROOT/'release-manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('release source/config changed: '+name)
    if digest(manifest['files'])!=manifest['build']:raise ValueError('invalid release manifest')
    return str(manifest['build'])


def compute_batch(matcher: Matcher, af: GPUFocus, bf: GPUFocus, first: int, end: int, width: int) -> dict[str,Any]:
    started=time.monotonic();scopes=[];hits=[];values=[];groups=[0]*len(matcher.filter_groups)
    counts={key:0 for key in ('focus_u','focus_v','cartesian_opportunities','valid_focused_candidates','genuine_cubes','local_survivors','cube_checks')}
    timings={key:0.0 for key in ('kernel_seconds','focus_enumeration_seconds','transfer_seconds','verification_seconds')}
    overflow_splits=[]

    def execute(tile: Tile) -> None:
        try:row=matcher.tile(tile,af,bf)
        except OverflowError:
            if tile.end-tile.first<2:raise
            overflow_splits.append(dict(first=str(tile.first),end=str(tile.end)))
            left,right=tile.split();execute(left);execute(right);return
        if row['status']!='PASS' or not row['host_verified'] or row['cpu_rejected_reports']:
            raise ArithmeticError('unverified GPU tile')
        for n in row['cpu_confirmed_hits']:
            witness=verify(int(n),matcher.config.p)
            if not witness['valid_pseudocube']:raise ArithmeticError('standalone candidate verification failed')
            hits.append(witness)
        scopes.append(dict(first=str(tile.first),end=str(tile.end),valid_focused_candidates=row['valid_focused_candidates'],
                           group_survivors=row['group_survivors'],local_survivors=row['local_survivors'],
                           output_sha256=digest(row['all_values'])))
        values.extend(row['all_values'])
        for key in counts:counts[key]+=int(row[key])
        for key in timings:timings[key]+=float(row[key])
        for i,value in enumerate(row['group_survivors']):groups[i]+=value
        notify('WATCHDOG=1')

    for start in range(first,end,width):execute(Tile(start,min(start+width,end)))
    if len(values)!=len(set(values)):raise ArithmeticError('duplicate output across disjoint subtiles')
    return dict(status='PASS',host_verified=True,**counts,**timings,subtile_scopes=scopes,
                filter_groups=matcher.filter_groups,group_survivors=groups,all_values=values,
                cpu_confirmed_hits=hits,overflow_splits=overflow_splits,
                elapsed_seconds=time.monotonic()-started,claim='Executed primary coverage; independent replay required before any minimum claim.')


def status(directory: Path, frontier: Frontier, runtime: Runtime, phase: str, **extra: Any) -> None:
    row=dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),frontier=frontier.state(),
             peak_device_pool_bytes=runtime.peak_device,**extra)
    durable(directory/'status.json',row)
    print(json.dumps(row,separators=(',',':')),flush=True)


def smoke(runtime: Runtime, directory: Path, config: Config, af: GPUFocus, bf: GPUFocus, build: str) -> None:
    parallel,serial=Matcher(runtime,config),Matcher(runtime,config,layout='serial')
    rows=[]
    for first in (0,100000000):
        tile=Tile(first,first+100000000)
        a,b=parallel.tile(tile,af,bf),serial.tile(tile,af,bf)
        for key in ('all_values','valid_focused_candidates','group_survivors','genuine_cubes'):
            if a[key]!=b[key]:raise ArithmeticError('production-geometry canary mismatch: '+key)
        rows.append(dict(first=str(first),parallel=a,serial=b))
    for p,n in ((613,674441580981249129037406633),(619,1000000007**3),(619,1000000009**3)):
        narrow=Config(p,n-1,n,A619,B619);v=narrow.canonical(n)[1]
        row=Matcher(runtime,narrow).tile(Tile(max(0,v-5000000),min(B619,v+5000001)),af,bf)
        if row['all_values']!=[str(n)]:raise ArithmeticError('production high-magnitude canary failed')
        rows.append(dict(canary=str(n),result=row))
    test=Frontier(directory/'canary-ledger',{'purpose':'production-canary'},build,200000000,100000000)
    job=test.next_batch();assert job
    result=compute_batch(parallel,af,bf,job[2],job[3],50000000)
    for key in ('all_values','valid_focused_candidates','group_survivors','genuine_cubes'):
        if result[key]!=rows[0]['parallel'][key]:raise ArithmeticError('subtile partition mismatch: '+key)
    test.complete(*job,result);test.db.close()
    test=Frontier(directory/'canary-ledger',{'purpose':'production-canary'},build,200000000,100000000)
    test.validate_last_receipt();job=test.next_batch();assert job and job[2]==100000000
    result=compute_batch(parallel,af,bf,job[2],job[3],50000000)
    for key in ('all_values','valid_focused_candidates','group_survivors','genuine_cubes'):
        if result[key]!=rows[1]['parallel'][key]:raise ArithmeticError('resumed subtile partition mismatch: '+key)
    test.complete(*job,result)
    audited=test.audit_production();test.db.close()
    full=compute_batch(parallel,af,bf,0,6400000000,100000000)
    if len(full['subtile_scopes'])<64:raise ArithmeticError('full production batch incomplete')
    durable(directory/'canary.json',dict(status='PASS',cases=rows,ledger=audited,full_batch=full,novel_production_coverage=0))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pseudocube_619_production.json')
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();raw,config=load_production(args.config);build=frozen_build()
    directory=args.state.resolve();directory.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=raw['device_uuid']:raise ValueError('single production device allocation required')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopping)
    gpu_lock=Path('/tmp')/('pseudocube-'+raw['device_uuid']+'.lock')
    with gpu_lock.open('a') as gpu_owner,(directory/'owner.lock').open('a') as state_owner:
        fcntl.flock(gpu_owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(state_owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        running=subprocess.check_output(['nvidia-smi','-i',raw['device_uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if running:raise RuntimeError('authorized GPU is occupied')
        runtime=Runtime(time.monotonic()+60 if args.smoke else float('inf'),raw['max_device_bytes'],raw['max_device_fraction'])
        if runtime.cp.cuda.runtime.getDeviceCount()!=1:raise RuntimeError('multi-GPU allocation rejected')
        af,bf=(GPUFocus(runtime,focus) for focus in config.focuses())
        if args.smoke:smoke(runtime,directory,config,af,bf,build);return
        frontier=Frontier(directory/'coverage',dict(production=raw,mathematics=config.manifest()),build,config.B,
                          raw['tile_width']*raw['tiles_per_commit'])
        frontier.validate_last_receipt()
        resumed=frontier.recover_stopped_owner('Exclusive GPU and state locks acquired by pid '+str(os.getpid())+'; previous owner no longer holds either lock')
        matcher=Matcher(runtime,config)
        durable(directory/'release.json',dict(build=build,source=str(ROOT),configuration=raw,mathematics=config.manifest(),compiles=runtime.compiles))
        status(directory,frontier,runtime,'running',recovered_attempts=resumed)
        notify('READY=1\nSTATUS=Searching pseudocube p=619 on one RTX 5080')
        last_report=0.0;last_resources=0.0
        try:
            while not STOP:
                now=time.monotonic()
                if now-last_resources>30:
                    used=int(subprocess.check_output(['du','-s','-B1',str(directory)],text=True).split()[0])
                    if used>=raw['max_state_bytes'] or shutil.disk_usage(directory).free<raw['min_free_disk_bytes']:
                        raise MemoryError('production storage guard reached')
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>raw['max_host_bytes']:
                        raise MemoryError('host memory guard reached')
                    last_resources=now
                job=frontier.next_batch()
                if job is None:break
                try:
                    result=compute_batch(matcher,af,bf,job[2],job[3],raw['tile_width'])
                    frontier.complete(*job,result)
                except Exception as exc:
                    frontier.fail(job[0],job[1],repr(exc));raise
                if now-last_report>15 or result['cpu_confirmed_hits']:
                    status(directory,frontier,runtime,'running',last_batch_seconds=result['elapsed_seconds'])
                    last_report=now
            status(directory,frontier,runtime,'stopped' if STOP else frontier.state()['campaign_state'])
        except Exception as exc:
            status(directory,frontier,runtime,'paused-error',error=repr(exc))
            raise
        finally:frontier.db.close()


if __name__=='__main__':
    try:main()
    except (MemoryError,ArithmeticError,ValueError,RuntimeError) as exc:
        print(json.dumps(dict(status='STOPPED_FOR_REVIEW',error=repr(exc))),flush=True);raise SystemExit(2)
    except BlockingIOError as exc:
        print(json.dumps(dict(status='ALLOCATION_BUSY',error=repr(exc))),flush=True);raise SystemExit(4)
