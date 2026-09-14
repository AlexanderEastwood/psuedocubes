#!/usr/bin/env python3
"""Bounded, isolated batch-cache and producer/consumer benchmarks; no production writes."""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import hashlib
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pseudocube.accounting import durable, digest
from pseudocube.batch_experiment import CachedMatcher, FusedMatcher, MaskedMatcher, Prepared, compare, prepare, prepare_arrays, tile_ranges
from pseudocube.focus import A619, B619, Config, Tile
from pseudocube.gpu import GPUFocus, Matcher, Runtime
from pseudocube.multi import DEVICES, SharedFrontier
from tools.run_production import compute_batch

WIDTH = 100000000
BATCH = WIDTH*64
CAP = 1000000


def config() -> Config:
    return Config(619, 10**27, 3*10**27, A619, B619)


def cached(matcher: CachedMatcher, af: GPUFocus, bf: GPUFocus, batch: Prepared) -> dict[str, Any]:
    matcher.prepared = batch
    result = compute_batch(matcher, af, bf, batch.first, batch.end, batch.width)
    result['elapsed_seconds'] += batch.prep_seconds
    result['focus_enumeration_seconds'] += batch.prep_seconds
    return result


def signature(result: dict[str, Any]) -> str:
    keys = ('focus_u','focus_v','cartesian_opportunities','valid_focused_candidates','genuine_cubes',
            'group_survivors','subtile_scopes','cpu_confirmed_hits')
    return digest(dict(**{key: result[key] for key in keys}, all_values=sorted(result['all_values'])))


class CPUFocus:
    def __init__(self, directory: Path, name: str, modulus: int) -> None:
        self.C = np.load(directory/(name+'-C.npy'), mmap_mode='r')
        self.D = np.load(directory/(name+'-D.npy'), mmap_mode='r')
        self.modulus = modulus
        lib = ctypes.CDLL(str(directory/'focus_window.so'))
        pointer = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags='C_CONTIGUOUS')
        self.fn = lib.focus_window
        self.fn.argtypes = [pointer, ctypes.c_uint64, pointer, ctypes.c_uint64,
                           ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                           pointer, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        self.fn.restype = ctypes.c_int

    def window(self, lo: int, hi: int, cap: int = CAP) -> np.ndarray:
        if not 0 <= lo <= hi < 1 << 64:
            raise ValueError('invalid CPU focus range')
        arrays = []
        count_all = 0
        for period in range(lo//self.modulus, (hi-1)//self.modulus+1):
            base = period*self.modulus
            out = np.empty(cap, dtype=np.uint64)
            count = ctypes.c_uint64()
            result = self.fn(self.C, len(self.C), self.D, len(self.D), self.modulus,
                             max(0, lo-base), min(self.modulus, hi-base), base, out, cap, ctypes.byref(count))
            if result == 1:
                raise OverflowError('CPU focus output overflow')
            if result:
                raise ValueError('invalid CPU focus arguments')
            count_all += count.value
            if count_all > cap:
                raise OverflowError('combined CPU focus overflow')
            arrays.append(out[:count.value])
        return np.sort(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.uint64)


def gpu_lock(device: str) -> Any:
    if device not in DEVICES:
        raise ValueError('unauthorized GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    lock = (Path('/tmp')/('pseudocube-'+device+'.lock')).open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+device)
    return lock


def single(output: Path, device: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    lock = gpu_lock(device)
    runtime = Runtime(time.monotonic()+120, 8*1024**3)
    c = config()
    af, bf = (GPUFocus(runtime, f) for f in c.focuses())
    baseline, cache, fused, masked = Matcher(runtime, c), CachedMatcher(runtime, c), FusedMatcher(runtime, c), MaskedMatcher(runtime,c)
    checks = []
    scopes = [(0, BATCH, WIDTH), (246112000000000-BATCH, 246112000000000, WIDTH),
              (B619//2, B619//2+BATCH, WIDTH), (B619-BATCH, B619, WIDTH)]
    # Reference tests preserve every per-tile population and output digest.
    for first, end, width in scopes:
        ref = compute_batch(baseline, af, bf, first, end, width)
        p = prepare(runtime, c, af, bf, first, end, width)
        cached_result = cached(cache, af, bf, p)
        fused_result = fused.run(p)
        masked_result = masked.run(p)
        compare(ref, cached_result)
        compare(ref, fused_result)
        compare(ref, masked_result)
        checks.append(dict(first=str(first), end=str(end), status='PASS', signature=signature(ref),
                           reference=ref, cached=cached_result, fused=fused_result,masked=masked_result))
    # Complete tractable sets and a partial final tile.
    for prime, A, B in [(7,14,9),(13,14,9*13),(19,14*19,9*13),(31,14*19,9*13*31)]:
        small = Config(prime, 0, 100000, A, B)
        sa, sb = (GPUFocus(runtime, f) for f in small.focuses())
        width = max(1, (B+6)//7)
        ref = compute_batch(Matcher(runtime, small), sa, sb, 0, B, width)
        p = prepare(runtime, small, sa, sb, 0, B, width)
        result = FusedMatcher(runtime, small).run(p)
        compare(ref, result)
        compare(ref,MaskedMatcher(runtime,small).run(p))
        from pseudocube.oracle import local3
        expected = {str(n) for n in range(1,100001) if local3(n,prime)}
        if set(result['all_values']) != expected:
            raise ArithmeticError('complete independent CPU output mismatch')
        overflow = False
        try:
            FusedMatcher(runtime, small).run(p, output_cap=1)
        except OverflowError:
            overflow = True
        if not overflow and prime == 7:
            raise ArithmeticError('expected overflowing output did not fail closed')
        if prime == 7:
            try:MaskedMatcher(runtime,small).run(p,output_cap=1)
            except OverflowError:pass
            else:raise ArithmeticError('masked output overflow did not fail closed')
        checks.append(dict(p=prime, status='PASS', complete_cpu_set=True, output_overflow_rejected=overflow))
    # Known high-magnitude noncube and genuine cubes, including candidate verification.
    for prime, n in [(613,674441580981249129037406633),(619,1000000007**3),(619,1000000009**3)]:
        narrow = Config(prime, n-1, n, A619, B619)
        v = narrow.canonical(n)[1]
        first, end = max(0,v-5000000), min(B619,v+5000001)
        p = prepare(runtime,narrow,af,bf,first,end,5000000)
        result = FusedMatcher(runtime,narrow).run(p)
        ref = compute_batch(Matcher(runtime,narrow),af,bf,first,end,5000000)
        compare(ref,result)
        compare(ref,MaskedMatcher(runtime,narrow).run(p))
        if result['all_values'] != [str(n)]:
            raise ArithmeticError('high-magnitude canary lost')
        checks.append(dict(p=prime,n=str(n),status='PASS'))
    cpu_dir = output/'cpu-focus'
    cpu_dir.mkdir()
    for name, f in [('A',af),('B',bf)]:
        np.save(cpu_dir/(name+'-C.npy'), f.C.get())
        np.save(cpu_dir/(name+'-D.npy'), np.sort(f.D.get()))
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'review/focus_window_cpu.cpp'),'-o',str(cpu_dir/'focus_window.so')],check=True)
    ca, cb = CPUFocus(cpu_dir,'A',A619), CPUFocus(cpu_dir,'B',B619)
    cpu_rows = []
    for first, end, _ in scopes:
        ranges = tile_ranges(c,first,end,WIDTH)
        start = time.perf_counter()
        us,vs=ca.window(ranges[0][2],ranges[-1][3]),cb.window(first,end)
        cpu_seconds=time.perf_counter()-start
        p=prepare(runtime,c,af,bf,first,end,WIDTH)
        if not np.array_equal(us,p.us.get()) or not np.array_equal(vs,p.vs.get()):
            raise ArithmeticError('CPU producer and GPU focus lists disagree')
        cpu_rows.append(dict(first=str(first),seconds=cpu_seconds,gpu_prepare_seconds=p.prep_seconds,coordinate_bytes=us.nbytes+vs.nbytes))
    repeats=[]
    for repeat in range(8):
        first=(repeat+50)*BATCH
        rows={}
        modes=['baseline','cached','fused','masked']
        for mode in modes[repeat%4:]+modes[:repeat%4]:
            start=time.perf_counter()
            if mode=='baseline':
                result=compute_batch(baseline,af,bf,first,first+BATCH,WIDTH)
            else:
                p=prepare(runtime,c,af,bf,first,first+BATCH,WIDTH)
                result=cached(cache,af,bf,p) if mode=='cached' else masked.run(p) if mode=='masked' else fused.run(p)
            rows[mode]=dict(wall_seconds=time.perf_counter()-start,prep_seconds=result['focus_enumeration_seconds'],
                            kernel_seconds=result['kernel_seconds'],prefilter_seconds=result.get('prefilter_build_seconds',0),
                            signature=signature(result),candidates=result['valid_focused_candidates'])
        if len({r['signature'] for r in rows.values()}) != 1:
            raise ArithmeticError('interleaved benchmark mismatch')
        repeats.append(rows)
    summary={mode:dict(median_seconds=statistics.median(r[mode]['wall_seconds'] for r in repeats),
                       median_prep_seconds=statistics.median(r[mode]['prep_seconds'] for r in repeats),
                       median_kernel_seconds=statistics.median(r[mode]['kernel_seconds'] for r in repeats),
                       median_prefilter_seconds=statistics.median(r[mode]['prefilter_seconds'] for r in repeats))
             for mode in ('baseline','cached','fused','masked')}
    result=dict(status='PASS',gpu=device,checks=checks,repeats=repeats,summary=summary,cpu_generation=cpu_rows,
                compiles=runtime.compiles,peak_device_bytes=runtime.peak_device,novel_production_coverage=0)
    durable(output/'single.json',result)
    print(json.dumps(dict(status='PASS',summary=summary,cpu_generation=cpu_rows,checks=len(checks))),flush=True)
    lock.close()


def runtime_worker(device: str) -> tuple[Any, Runtime]:
    lock=gpu_lock(device)
    return lock, Runtime(time.monotonic()+120,8*1024**3)


def fleet_worker(device: str, mode: str, indices: list[int], ready: Any, start: Any, results: Any,
                 ledger: Path | None = None, jobs: int = 192) -> None:
    try:
        lock,runtime=runtime_worker(device)
        c=config();af,bf=(GPUFocus(runtime,f) for f in c.focuses())
        matcher=Matcher(runtime,c) if mode=='baseline' else CachedMatcher(runtime,c) if mode=='cached' else MaskedMatcher(runtime,c) if mode=='masked' else FusedMatcher(runtime,c)
        frontier=SharedFrontier(ledger,dict(purpose='durable-validation',mathematics=c.manifest(),jobs=jobs),
                                 'isolated-batch-experiment-v3',jobs*BATCH,BATCH,DEVICES) if ledger else None
        # Warm one validation job, excluded from timed work and still no production credit.
        p=prepare(runtime,c,af,bf,0,BATCH,WIDTH)
        if isinstance(matcher,FusedMatcher):matcher.run(p)
        elif isinstance(matcher,CachedMatcher):cached(matcher,af,bf,p)
        else:compute_batch(matcher,af,bf,0,BATCH,WIDTH)
        ready.put(device);start.wait(60)
        begun=time.perf_counter();rows=[]
        iterator=iter(indices)
        while True:
            job=None
            if frontier:
                job=frontier.next_batch(device)
                if job is None:break
                first=job[2];index=first//BATCH
            else:
                try:index=next(iterator)
                except StopIteration:break
                first=index*BATCH
            if isinstance(matcher,FusedMatcher):
                p=prepare(runtime,c,af,bf,first,first+BATCH,WIDTH)
                row=matcher.run(p)
            elif isinstance(matcher,CachedMatcher):
                p=prepare(runtime,c,af,bf,first,first+BATCH,WIDTH)
                row=cached(matcher,af,bf,p)
            else:row=compute_batch(matcher,af,bf,first,first+BATCH,WIDTH)
            if frontier and job:frontier.complete(device,job,row)
            rows.append(dict(job=index,signature=signature(row),candidates=row['valid_focused_candidates']))
        results.put(dict(device=device,rows=rows,seconds=time.perf_counter()-begun,status='PASS'))
        if frontier:frontier.db.close()
        lock.close()
    except BaseException as exc:
        results.put(dict(device=device,status='FAIL',error=repr(exc)))
        raise


def pipeline_producer(mode: str, device: str, names: list[str], slots: Any, queues: list[Any],
                      jobs: int, cpu_dir: Path, ready: Any, start: Any, results: Any) -> None:
    attached=[]
    try:
        attached=[shared_memory.SharedMemory(name=name) for name in names]
        buffers=[np.ndarray((2*CAP,),dtype=np.uint64,buffer=mem.buf) for mem in attached]
        c=config();lock=None;runtime=None
        if mode.endswith('gpu-producer'):
            lock,runtime=runtime_worker(device)
            af,bf=(GPUFocus(runtime,f) for f in c.focuses())
        else:
            af,bf=CPUFocus(cpu_dir,'A',A619),CPUFocus(cpu_dir,'B',B619)
        ready.put('producer');start.wait(60)
        begun=time.perf_counter();generation=transport=waiting=0.0
        super_us: Any=None
        super_vs: Any=None
        for index in range(jobs):
            first=index*BATCH;ranges=tile_ranges(c,first,first+BATCH,WIDTH)
            t=time.perf_counter()
            owner,slot=slots.get(timeout=60)
            waiting+=time.perf_counter()-t;t=time.perf_counter()
            # Amortize the producer's partial-array scan across eight jobs.
            if index % 8 == 0:
                super_end=min(index+8,jobs)*BATCH
                super_us=af.window(c.u_interval(first)[0],c.u_interval(super_end-1)[1])
                super_vs=bf.window(first,super_end)
                if runtime is not None:
                    super_us,super_vs=super_us.get(),super_vs.get()
            ua,ub=np.searchsorted(super_us,[ranges[0][2],ranges[-1][3]])
            va,vb=np.searchsorted(super_vs,[first,first+BATCH])
            us,vs=super_us[ua:ub],super_vs[va:vb]
            generation+=time.perf_counter()-t;t=time.perf_counter()
            target=buffers[slot]
            target[:len(us)]=us;target[CAP:CAP+len(vs)]=vs
            sha=hashlib.sha256(us.tobytes()+vs.tobytes()).hexdigest()
            queues[owner].put((index,slot,len(us),len(vs),sha))
            transport+=time.perf_counter()-t
        for q in queues:q.put(None)
        results.put(dict(role='producer',status='PASS',seconds=time.perf_counter()-begun,generation_seconds=generation,
                         staging_seconds=transport,backpressure_seconds=waiting,jobs=jobs))
        if lock:lock.close()
    except BaseException as exc:
        results.put(dict(role='producer',status='FAIL',error=repr(exc)))
        raise
    finally:
        for mem in attached:mem.close()


def pipeline_consumer(device: str, owner: int, names: list[str], slots: Any, tasks: Any,
                      ready: Any, start: Any, results: Any, masked: bool = False) -> None:
    attached=[]
    try:
        lock,runtime=runtime_worker(device);c=config();matcher=MaskedMatcher(runtime,c) if masked else FusedMatcher(runtime,c)
        attached=[shared_memory.SharedMemory(name=name) for name in names]
        buffers=[np.ndarray((2*CAP,),dtype=np.uint64,buffer=mem.buf) for mem in attached]
        ready.put(device);start.wait(60)
        begun=time.perf_counter();rows=[];waiting=upload=filtering=0.0
        while True:
            t=time.perf_counter();task=tasks.get(timeout=60);waiting+=time.perf_counter()-t
            if task is None:break
            index,slot,nu,nv,sha=task
            source=buffers[slot]
            t=time.perf_counter()
            if hashlib.sha256(source[:nu].tobytes()+source[CAP:CAP+nv].tobytes()).hexdigest()!=sha:
                raise ArithmeticError('producer/consumer coordinate checksum mismatch')
            us=runtime.cp.asarray(source[:nu]);vs=runtime.cp.asarray(source[CAP:CAP+nv])
            first=index*BATCH
            p=prepare_arrays(runtime,c,first,first+BATCH,WIDTH,us,vs,t)
            upload+=time.perf_counter()-t
            # The device owns a completed copy before this host slot is reusable.
            slots.put((owner,slot))
            t=time.perf_counter();row=matcher.run(p);filtering+=time.perf_counter()-t
            rows.append(dict(job=index,signature=signature(row),candidates=row['valid_focused_candidates']))
        results.put(dict(role='consumer',device=device,status='PASS',seconds=time.perf_counter()-begun,
                         waiting_seconds=waiting,upload_seconds=upload,filter_seconds=filtering,rows=rows))
        lock.close()
    except BaseException as exc:
        results.put(dict(role='consumer',device=device,status='FAIL',error=repr(exc)))
        raise
    finally:
        for mem in attached:mem.close()


def fleet(output: Path, mode: str, jobs: int, cpu_dir: Path) -> None:
    maximum=8192 if mode=='durable-masked' else 256
    if not 4 <= jobs <= maximum:
        raise ValueError('job count exceeds the bounded experiment allocation')
    output.mkdir(parents=True,exist_ok=False)
    ctx=mp.get_context('spawn');ready=ctx.Queue();results=ctx.Queue();start=ctx.Event()
    processes=[];memories=[]
    ledger=None
    if mode.startswith('durable-'):
        ledger=output/'coverage'
        initial=SharedFrontier(ledger,dict(purpose='durable-validation',mathematics=config().manifest(),jobs=jobs),
                               'isolated-batch-experiment-v3',jobs*BATCH,BATCH,DEVICES,
                               dict(cursor='0',commits='0',candidates='0',cubes='0',subtiles='0',campaign_state='searching'))
        initial.db.close()
    if mode in ('baseline','cached','fused','masked') or ledger:
        for i,device in enumerate(DEVICES):
            processes.append(ctx.Process(target=fleet_worker,args=(device,mode.removeprefix('durable-'),list(range(i,jobs,4)),ready,start,results,ledger,jobs)))
    else:
        devices=DEVICES[1:] if mode.endswith('gpu-producer') else DEVICES
        slots=ctx.Queue();queues=[ctx.Queue(maxsize=2) for _ in devices]
        for i in range(len(devices)):
            for _ in range(2):
                mem=shared_memory.SharedMemory(create=True,size=2*CAP*8)
                slot=len(memories);memories.append(mem);slots.put((i,slot))
        names=[mem.name for mem in memories]
        for i,device in enumerate(devices):
            processes.append(ctx.Process(target=pipeline_consumer,args=(device,i,names,slots,queues[i],ready,start,results,mode.startswith('masked-'))))
        processes.append(ctx.Process(target=pipeline_producer,args=(mode,DEVICES[0],names,slots,queues,jobs,cpu_dir,ready,start,results)))
    try:
        for process in processes:process.start()
        for _ in processes:ready.get(timeout=60)
        begun=time.perf_counter();start.set()
        collected=[results.get(timeout=90) for _ in processes]
        elapsed=time.perf_counter()-begun
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:raise RuntimeError('isolated benchmark worker failed')
        if any(row['status']!='PASS' for row in collected):raise RuntimeError(str(collected))
        rows=sorted([row for worker in collected for row in worker.get('rows',[])],key=lambda row:row['job'])
        if [row['job'] for row in rows] != list(range(jobs)):raise ArithmeticError('missing or duplicate benchmark jobs')
        result: dict[str,Any]=dict(status='PASS',mode=mode,jobs=jobs,elapsed_seconds=elapsed,workers=collected,rows=rows,
                    candidates=sum(row['candidates'] for row in rows),novel_production_coverage=0,
                    production_services_started=False)
        result['candidates_per_second']=result['candidates']/elapsed
        if ledger:
            audit=SharedFrontier(ledger,dict(purpose='durable-validation',mathematics=config().manifest(),jobs=jobs),
                                 'isolated-batch-experiment-v3',jobs*BATCH,BATCH,DEVICES)
            result['ledger_audit']=audit.audit_shared();audit.db.close()
            result['ledger_scope']=f'Only the {jobs}-job validation interval; zero novel production coverage.'
        durable(output/'fleet.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('workers','rows')}),flush=True)
    finally:
        # These are only child processes created by this isolated test harness.
        for process in processes:
            if process.is_alive():process.terminate();process.join(timeout=10)
        for mem in memories:mem.close();mem.unlink()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['single','baseline','cached','fused','masked','gpu-producer','cpu-producer',
                                      'masked-gpu-producer','masked-cpu-producer','durable-baseline','durable-fused','durable-masked'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default=DEVICES[0],choices=DEVICES)
    parser.add_argument('--jobs',type=int,default=192)
    parser.add_argument('--cpu-focus',type=Path,default=Path('/nonexistent'))
    args=parser.parse_args()
    if args.mode=='single':single(args.output,args.device)
    else:fleet(args.output,args.mode,args.jobs,args.cpu_focus)


if __name__=='__main__':main()
