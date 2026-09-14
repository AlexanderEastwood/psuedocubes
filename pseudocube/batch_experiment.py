"""Batched CRT engines; the qualified bitset path is used by optimized production."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .accounting import digest
from .focus import Config, Tile
from .gpu import GPUFocus, Matcher, Runtime, MASK, WIDE, match_source, launch_plan
from .oracle import is_perfect_cube, local3
from review.verify_cube import verify


@dataclass
class Prepared:
    first: int
    end: int
    width: int
    us: Any
    vs: Any
    bounds: list[tuple[int, int, int, int]]
    prep_seconds: float


def tile_ranges(config: Config, first: int, end: int, width: int) -> list[tuple[int, int, int, int]]:
    Tile(first, end).validate(config)
    if width <= 0 or (end-first+width-1)//width > 64:
        raise ValueError('experiment allows at most 64 bounded subtiles')
    return [(v, min(v+width, end), config.u_interval(v)[0], config.u_interval(min(v+width, end)-1)[1])
            for v in range(first, end, width)]


def prepare_arrays(runtime: Runtime, config: Config, first: int, end: int, width: int,
                   us: Any, vs: Any, started: float) -> Prepared:
    ranges = tile_ranges(config, first, end, width)
    cp = runtime.cp
    u_queries = np.array([bound for _, _, lo, hi in ranges for bound in (lo, hi)], dtype=np.uint64)
    v_queries = np.array([bound for lo, hi, _, _ in ranges for bound in (lo, hi)], dtype=np.uint64)
    ub = cp.searchsorted(us, cp.asarray(u_queries)).get().reshape(-1, 2)
    vb = cp.searchsorted(vs, cp.asarray(v_queries)).get().reshape(-1, 2)
    bounds = [(int(a), int(b), int(c), int(d)) for (a, b), (c, d) in zip(ub, vb)]
    for a, b, c, d in bounds:
        if (b-a)*(d-c) > 50_000_000:
            raise OverflowError('per-subtile candidate opportunity limit exceeded')
    runtime.check()
    return Prepared(first, end, width, us, vs, bounds, time.perf_counter()-started)


def prepare(runtime: Runtime, config: Config, af: GPUFocus, bf: GPUFocus,
            first: int, end: int, width: int = 100000000) -> Prepared:
    started = time.perf_counter()
    ranges = tile_ranges(config, first, end, width)
    us = af.window(ranges[0][2], ranges[-1][3])
    vs = bf.window(first, end)
    return prepare_arrays(runtime, config, first, end, width, us, vs, started)


class CachedMatcher(Matcher):
    prepared: Prepared | None = None

    def tile(self, tile: Tile, af: GPUFocus, bf: GPUFocus, cap: int = 1000000, output_cap: int = 8192) -> dict[str, Any]:
        started = time.perf_counter()
        batch = self.prepared
        if batch is None or not batch.first <= tile.first < tile.end <= batch.end:
            raise ValueError('tile outside prepared batch')
        index, remainder = divmod(tile.first-batch.first, batch.width)
        if remainder or tile.end != min(tile.first+batch.width, batch.end):
            raise ValueError('unexpected prepared subtile geometry')
        a, b, c, d = batch.bounds[index]
        result = self.arrays(batch.us[a:b], batch.vs[c:d], output_cap)
        result['focus_enumeration_seconds'] = 0.0
        result['tile'] = tile.manifest(self.config)
        result['end_to_end_seconds'] = time.perf_counter()-started
        return result


class FusedMatcher:
    """Same generated arithmetic, with separate counters/output capacity per tile."""
    def __init__(self, runtime: Runtime, config: Config) -> None:
        self.runtime, self.config, self.cp = runtime, config, runtime.cp
        source, table, _ = match_source(config, 'byte', 'warp')
        residuals = config.residuals()
        residuals.sort(key=lambda q: (q != 9 and q % 3 != 1, q))
        self.filter_groups = [residuals[i:i+4] for i in range(0, len(residuals), 4)]
        ng = len(self.filter_groups)
        replacements = {
            'unsigned long long* counts,unsigned long long* out,': 'unsigned long long* counters,unsigned long long* outputs,',
            'unsigned long long cap,unsigned long long stripes){': 'unsigned long long cap,unsigned long long stripes,unsigned long long v_first,unsigned long long tile_width){',
            'U shift=mul(vs[j],': f'unsigned long long tile_id=(vs[j]-v_first)/tile_width;\nunsigned long long* counts=counters+tile_id*{ng+3};\nunsigned long long* out=outputs+tile_id*2*cap;\nU shift=mul(vs[j],',
        }
        for old, new in replacements.items():
            if source.count(old) != 1:
                raise ValueError('review required: reference kernel structure changed')
            source = source.replace(old, new)
        self.kernel = runtime.compile(source, 'match')
        self.table = self.cp.asarray(np.array(table, dtype=np.uint8))
        self.sm_count = int(self.cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount'])

    def extra_args(self, batch: Prepared) -> tuple[Any, ...]:
        return ()

    def run(self, batch: Prepared, output_cap: int = 8192) -> dict[str, Any]:
        started = time.perf_counter()
        ranges = tile_ranges(self.config, batch.first, batch.end, batch.width)
        if len(ranges) != len(batch.bounds) or not 0 < output_cap <= 8192:
            raise ValueError('invalid prepared geometry or output capacity')
        ng = len(self.filter_groups)
        counts = self.cp.zeros((len(ranges), ng+3), dtype=self.cp.uint64)
        outputs = self.cp.empty((len(ranges), output_cap*2), dtype=self.cp.uint64)
        blocks, stripes = launch_plan(int(batch.us.size), int(batch.vs.size), self.sm_count, 'warp', 2)
        extra_args = self.extra_args(batch)
        kernel_seconds = 0.0
        if blocks:
            c = self.config
            kernel_seconds = self.runtime.measured(self.kernel, (blocks,),
                (batch.us, np.uint64(batch.us.size), batch.vs, np.uint64(batch.vs.size),
                 np.uint64(c.lo & MASK), np.uint64(c.lo >> 64), np.uint64(c.hi & MASK), np.uint64(c.hi >> 64),
                 self.table, counts, outputs, np.uint64(output_cap), np.uint64(stripes), np.uint64(batch.first), np.uint64(batch.width)) + extra_args)
        transfer_started = time.perf_counter()
        populations = counts.get()
        if np.any(populations[:, ng+2]):
            raise ArithmeticError('unsigned underflow')
        if np.any(populations[:, ng+1] > output_cap):
            raise OverflowError('fused tile output overflow; no truncated coverage')
        indices = [row*output_cap*2+offset for row in range(len(ranges)) for offset in range(2*int(populations[row, ng+1]))]
        limbs = outputs.ravel()[self.cp.asarray(np.array(indices, dtype=np.int64))].get().reshape(-1, 2) if indices else []
        raw_values = [int(a)+(int(b) << 64) for a, b in limbs]
        all_values: list[int] = []
        offset = 0
        for row in range(len(ranges)):
            count = int(populations[row, ng+1])
            all_values.extend(sorted(raw_values[offset:offset+count]))
            offset += count
        transfer_seconds = time.perf_counter()-transfer_started
        verification_started = time.perf_counter()
        if len(all_values) != len(set(all_values)):
            raise ArithmeticError('duplicate canonical output')
        if any(not self.config.lo < n <= self.config.hi or not local3(n, self.config.p) for n in all_values):
            raise ArithmeticError('GPU/CPU disagreement')
        hits = [verify(n, self.config.p) for n in all_values if not is_perfect_cube(n)]
        if any(not row['valid_pseudocube'] for row in hits):
            raise ArithmeticError('independent witness verification failed')
        scopes = []
        position = 0
        for row, (first, end, _, _) in enumerate(ranges):
            number = int(populations[row, ng+1])
            values = sorted(all_values[position:position+number])
            if any(not first <= self.config.canonical(n)[1] < end for n in values):
                raise ArithmeticError('output assigned to wrong subtile')
            scopes.append(dict(first=str(first), end=str(end), valid_focused_candidates=int(populations[row, 0]),
                               group_survivors=[int(n) for n in populations[row, 1:ng+1]], local_survivors=number,
                               output_sha256=digest([str(n) for n in values])))
            position += number
        verification_seconds = time.perf_counter()-verification_started
        groups = [int(n) for n in populations[:, 1:ng+1].sum(axis=0)]
        return dict(status='PASS', host_verified=True, focus_u=sum(b-a for a, b, _, _ in batch.bounds),
                    focus_v=int(batch.vs.size), cartesian_opportunities=sum((b-a)*(d-c) for a, b, c, d in batch.bounds),
                    valid_focused_candidates=sum(int(n) for n in populations[:, 0]),
                    genuine_cubes=len(all_values)-len(hits), local_survivors=len(all_values), cube_checks=len(all_values),
                    kernel_seconds=kernel_seconds, transfer_seconds=transfer_seconds, verification_seconds=verification_seconds,
                    focus_enumeration_seconds=batch.prep_seconds, subtile_scopes=scopes, filter_groups=self.filter_groups,
                    group_survivors=groups, all_values=[str(n) for n in all_values], cpu_confirmed_hits=hits,
                    overflow_splits=[], elapsed_seconds=time.perf_counter()-started+batch.prep_seconds,
                    kernel_registers=int(self.kernel.num_regs), kernel_local_bytes=int(self.kernel.local_size_bytes),
                    claim='Isolated validation only; no production coverage.', matching_layout='fused-batch')


class MaskedMatcher(FusedMatcher):
    """Intersect first-group residue bitsets before constructing wide candidates."""
    def __init__(self, runtime: Runtime, config: Config) -> None:
        super().__init__(runtime, config)
        from .tables import allowed
        residuals=[q for group in self.filter_groups for q in group]
        if not residuals:
            raise ValueError('bitset experiment requires at least one residual filter')
        first=residuals[:4];ng=len(self.filter_groups)
        offsets=[];offset=0
        for q in residuals:
            offsets.append(offset);offset+=q
        self.mask_kernels=[];self.mask_rows=sum(first);row_offset=0
        pointers=[];intersections=[]
        for index,q in enumerate(first):
            source=f'''extern "C" __global__ void mask_build(const unsigned long long* us,unsigned long long nu,
const unsigned char* allowed,unsigned int* masks,unsigned long long words){{
unsigned long long t=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x,warp=t/32;
unsigned int lane=threadIdx.x%32;unsigned long long r=warp/words,word=warp%words;
if(r>={q})return;unsigned long long i=word*32+lane;bool ok=false;
if(i<nu){{unsigned long long m=((us[i]%{q}ULL)*{config.B%q}ULL+{q}ULL-(r*{config.A%q}ULL)%{q}ULL)%{q}ULL;
ok=allowed[{offsets[index]}ULL+m]!=0;}}
unsigned int bits=__ballot_sync(0xffffffffU,ok);
if(lane==0)masks[({row_offset}ULL+r)*words+word]=bits;}}'''
            self.mask_kernels.append((q,runtime.compile(source,'mask_build')))
            pointers.append(f'const unsigned int* m{index}=masks+({row_offset}ULL+vs[j]%{q}ULL)*words;')
            intersections.append(f'bits&=m{index}[word];')
            row_offset+=q
        code=[WIDE,'''extern "C" __global__ void match(const unsigned long long* us,unsigned long long nu,
const unsigned long long* vs,unsigned long long nv,unsigned long long lo0,unsigned long long lo1,
unsigned long long hi0,unsigned long long hi1,const unsigned char* allowed,
unsigned long long* counters,unsigned long long* outputs,unsigned long long cap,unsigned long long stripes,
unsigned long long v_first,unsigned long long tile_width,const unsigned int* masks,unsigned long long words){
unsigned long long warp=(blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x)/32;
if(warp>=nv*stripes)return;unsigned int lane=threadIdx.x%32;
unsigned long long j=warp/stripes,stripe=warp%stripes;
unsigned long long tile_id=(vs[j]-v_first)/tile_width;''',
              f'''unsigned long long* counts=counters+tile_id*{ng+3};unsigned long long* out=outputs+tile_id*2*cap;
unsigned long long pop[{ng+1}]={{0}};
U shift=mul(vs[j],{config.A}ULL),lower=add(shift,U{{lo0,lo1}}),upper=add(shift,U{{hi0,hi1}});
unsigned long long left=0,h=nu;
while(left<h){{unsigned long long mid=left+(h-left)/2;
if(!less(lower,mul(us[mid],{config.B}ULL)))left=mid+1;else h=mid;}}
unsigned long long right=left;h=nu;
while(right<h){{unsigned long long mid=right+(h-right)/2;
if(!less(upper,mul(us[mid],{config.B}ULL)))right=mid+1;else h=mid;}}
''',*pointers,
              '''for(unsigned long long word=left/32+stripe*32+lane;word<(right+31)/32;word+=stripes*32){
unsigned long long base=word*32;unsigned int bits=0xffffffffU;
if(base<left)bits&=0xffffffffU<<(left-base);
if(right<base+32)bits&=(1U<<(right-base))-1U;
pop[0]+=__popc(bits);''',*intersections,
              f'''while(bits){{unsigned int bit=__ffs(bits)-1;bits&=bits-1;
U positive=mul(us[base+bit],{config.B}ULL);
if(less(positive,shift)){{atomicAdd(counts+{ng+2},1ULL);continue;}}
U x=sub(positive,shift);++pop[1];''']
        for index,q in enumerate(residuals[4:],start=4):
            code.append(f'{{unsigned int m=((x.hi%{q}ULL)*{(1<<64)%q}ULL+x.lo%{q}ULL)%{q}ULL;if(!allowed[{offsets[index]}ULL+m])continue;}}')
            if index%4==3 or index+1==len(residuals):code.append(f'++pop[{index//4+1}];')
        code.append(f'''unsigned long long slot=atomicAdd(counts+{ng+1},1ULL);
if(slot<cap){{out[2*slot]=x.lo;out[2*slot+1]=x.hi;}}
}}}}
for(unsigned int g=0;g<{ng+1};++g){{unsigned long long total=pop[g];
for(unsigned int delta=16;delta;delta>>=1)total+=__shfl_down_sync(0xffffffffU,total,delta);
if(lane==0 && total)atomicAdd(counts+g,total);}}}}''')
        self.kernel=runtime.compile('\n'.join(code),'match')
        self.masks: Any=None
        self.mask_seconds=0.0

    def extra_args(self, batch: Prepared) -> tuple[Any, ...]:
        words=(int(batch.us.size)+31)//32
        self.masks=self.cp.empty((self.mask_rows,max(1,words)),dtype=self.cp.uint32)
        started=time.perf_counter()
        if words:
            for q,kernel in self.mask_kernels:
                self.runtime.measured(kernel,((q*words+7)//8,),
                    (batch.us,np.uint64(batch.us.size),self.table,self.masks,np.uint64(words)))
        self.mask_seconds=time.perf_counter()-started
        return self.masks,np.uint64(words)

    def run(self, batch: Prepared, output_cap: int = 8192) -> dict[str, Any]:
        result=super().run(batch,output_cap)
        result['matching_layout']='residue-bitset-batch'
        result['prefilter_build_seconds']=self.mask_seconds
        return result


def compare(reference: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in ('focus_u', 'focus_v', 'cartesian_opportunities', 'valid_focused_candidates', 'genuine_cubes',
                'local_survivors', 'cube_checks', 'filter_groups', 'group_survivors', 'subtile_scopes', 'cpu_confirmed_hits'):
        if reference[key] != actual[key]:
            raise ArithmeticError('batch/reference mismatch: '+key)
    if sorted(reference['all_values']) != sorted(actual['all_values']):
        raise ArithmeticError('complete output mismatch')
