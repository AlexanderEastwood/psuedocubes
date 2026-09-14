"""CuPy/NVRTC baseline adapted from the pseudosquare factored-window structure."""
from __future__ import annotations
import hashlib
import importlib
from math import prod
from pathlib import Path
import time
from typing import Any

import numpy as np

from .focus import Config, Focus, Tile
from .oracle import is_perfect_cube, local3
from .tables import allowed, packed

ROOT=Path(__file__).resolve().parents[1]
MASK=(1<<64)-1
WIDE=(ROOT/'kernels/pseudocube.cuh').read_text()
RANGE=r'''
__device__ unsigned long long lb(const unsigned long long* a,unsigned long long n,unsigned long long v){
 unsigned long long l=0,h=n;while(l<h){unsigned long long m=l+(h-l)/2;if(a[m]<v)l=m+1;else h=m;}return l;}
extern "C" __global__ void window(const unsigned long long* C,unsigned long long nc,
 const unsigned long long* D,unsigned long long nd,unsigned long long M,
 unsigned long long lo,unsigned long long hi,unsigned long long offset,
 unsigned long long* out,unsigned long long* count,unsigned long long cap){
 unsigned long long j=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;if(j>=nd)return;
 unsigned long long d=D[j];for(unsigned int wrap=0;wrap<2;++wrap){
  unsigned long long a=lo+(wrap?M:0),b=hi+(wrap?M:0);
  a=a>d?a-d:0;b=b>d?b-d:0;if(b>M)b=M;if(a>=b)continue;
  unsigned long long begin=lb(C,nc,a),end=lb(C,nc,b);if(begin==end)continue;
  unsigned long long slot=atomicAdd(count,end-begin);
  for(unsigned long long i=begin;i<end;++i){unsigned long long pos=slot+i-begin;
   if(pos<cap)out[pos]=C[i]+d-(wrap?M:0)+offset;}
 }
}'''


class Runtime:
    def __init__(self, deadline: float, memory_cap: int, memory_fraction: float = 0.5) -> None:
        self.cp: Any=importlib.import_module('cupy')
        self.deadline=deadline
        free,total=self.cp.cuda.runtime.memGetInfo()
        self.cap=min(memory_cap,int(total*memory_fraction),int(free)-256*1024**2)
        if self.cap<=0:raise MemoryError('no permitted device memory')
        self.cp.get_default_memory_pool().set_limit(size=self.cap)
        self.compiles: list[dict[str,Any]]=[]
        self.peak_device=0

    def check(self) -> None:
        if time.monotonic()>=self.deadline-3:raise TimeoutError('BUDGET_EXHAUSTED')
        self.peak_device=max(self.peak_device,int(self.cp.get_default_memory_pool().total_bytes()))
        if self.peak_device>self.cap:raise MemoryError('device memory cap exceeded')

    def compile(self, source: str, name: str) -> Any:
        self.check();started=time.perf_counter()
        kernel=self.cp.RawKernel(source,name,options=('-std=c++17',),backend='nvrtc')
        kernel.compile()
        self.compiles.append(dict(name=name,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                 wall_seconds=time.perf_counter()-started,registers=kernel.num_regs,
                                 local_bytes=kernel.local_size_bytes,shared_bytes=kernel.shared_size_bytes))
        self.check();return kernel

    def measured(self, kernel: Any, grid: tuple[int,...], args: tuple[Any,...]) -> float:
        self.check()
        a,b=self.cp.cuda.Event(),self.cp.cuda.Event();a.record()
        kernel(grid,(256,),args);b.record();b.synchronize()
        self.check();return float(self.cp.cuda.get_elapsed_time(a,b))/1000


class GPUFocus:
    def __init__(self, runtime: Runtime, focus: Focus) -> None:
        self.runtime,self.focus=runtime,focus
        self.cp=runtime.cp
        first,second=focus.halves()
        self.half_sizes=[prod(map(len,first)),prod(map(len,second))]
        # Account for two output arrays plus sorting scratch, not the full CRT list.
        if sum(self.half_sizes)*32>runtime.cap//2:raise MemoryError('focus halves exceed pilot budget')
        self.C=self.cp.sort(self.partial(first));self.D=self.partial(second)
        self.query=runtime.compile(RANGE,'window')
        self.cp.cuda.Stream.null.synchronize()

    def partial(self, rings: tuple[tuple[int,...],...]) -> Any:
        terms=[];lines=['extern "C" __global__ void partial(unsigned long long n,const unsigned long long* terms,unsigned long long* out){unsigned long long i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;if(i>=n)return;unsigned long long r=0,rem=i;']
        for ring in rings:
            offset=len(terms);terms.extend(ring)
            lines.append(f'r+=terms[{offset}ULL+rem%{len(ring)}ULL];rem/={len(ring)}ULL;if(r>={self.focus.modulus}ULL)r-={self.focus.modulus}ULL;')
        lines.append('out[i]=r;}')
        count=prod(map(len,rings))
        data=self.cp.asarray(np.asarray(terms or [0],dtype=np.uint64));out=self.cp.empty(count,dtype=self.cp.uint64)
        kernel=self.runtime.compile('\n'.join(lines),'partial')
        self.runtime.measured(kernel,((count+255)//256,),(np.uint64(count),data,out))
        return out

    def window(self, low: int, high: int, cap: int = 1000000) -> Any:
        if not 0<=low<=high<1<<64:raise ValueError('invalid bounded focus window')
        if high==low:return self.cp.empty(0,dtype=self.cp.uint64)
        if (high-1)//self.focus.modulus-low//self.focus.modulus+1>4096:
            raise OverflowError('focus window exceeds bounded period count')
        result=[]
        allocation=min(cap,self.focus.count)
        for period in range(low//self.focus.modulus,(high-1)//self.focus.modulus+1):
            base=period*self.focus.modulus
            a,b=max(0,low-base),min(self.focus.modulus,high-base)
            out=self.cp.empty(allocation,dtype=self.cp.uint64);count=self.cp.zeros(1,dtype=self.cp.uint64)
            self.runtime.measured(self.query,((int(self.D.size)+255)//256,),
                (self.C,np.uint64(self.C.size),self.D,np.uint64(self.D.size),np.uint64(self.focus.modulus),
                 np.uint64(a),np.uint64(b),np.uint64(base),out,count,np.uint64(allocation)))
            length=int(count.get()[0])
            if length>allocation or sum(int(x.size) for x in result)+length>cap:
                raise OverflowError('focus buffer overflow; tile invalid')
            result.append(out[:length])
        merged=self.cp.concatenate(result)
        if merged.size>cap:raise OverflowError('combined focus buffer overflow')
        return self.cp.sort(merged)


def launch_plan(nu: int, nv: int, sm_count: int, layout: str, blocks_per_sm: int = 2) -> tuple[int,int]:
    """Return (blocks, warps per v row); partition positions, never the predicate."""
    if layout not in ('serial','warp') or type(blocks_per_sm) is not int or blocks_per_sm not in (1,2,4):
        raise ValueError('invalid matching layout')
    if min(nu,nv)<0 or sm_count<1:raise ValueError('invalid launch dimensions')
    if not nu or not nv:return 0,1
    if layout=='serial':return (nv+255)//256,1
    # Eight warps per 256-thread block. Keep small rows bounded by their work.
    stripes=min((nu+31)//32,max(1,(sm_count*blocks_per_sm*8+nv-1)//nv))
    return (nv*stripes+7)//8,stripes


def match_source(config: Config, variant: str, layout: str = 'warp') -> tuple[str,list[int],str]:
    if variant not in ('byte','packed','constant'):raise ValueError('unknown lookup variant')
    if layout not in ('serial','warp'):raise ValueError('unknown matching layout')
    residuals=config.residuals()
    # Enforce cubic constraints first; all coprimality-only primes still remain.
    residuals.sort(key=lambda q:(q!=9 and q%3!=1,q))
    table=[];offsets=[]
    for q in residuals:
        offsets.append(len(table))
        if variant=='byte':
            ring=[0]*q
            for a in allowed(q):ring[a]=1
            table.extend(ring)
        else:table.extend(packed(q))
    groups=(len(residuals)+3)//4
    dtype='unsigned char' if variant=='byte' else 'unsigned int'
    code=[WIDE]
    if variant=='constant':code.append(f'__device__ __constant__ unsigned int frozen[{max(1,len(table))}]={{'+','.join(map(str,table or [0]))+'};')
    code.append(f'''extern "C" __global__ void match(const unsigned long long* us,unsigned long long nu,
const unsigned long long* vs,unsigned long long nv,unsigned long long lo0,unsigned long long lo1,
unsigned long long hi0,unsigned long long hi1,const {dtype}* allowed,
unsigned long long* counts,unsigned long long* out,unsigned long long cap,unsigned long long stripes){{''')
    if layout=='serial':
        code.append('unsigned long long j=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;if(j>=nv)return;unsigned long long offset=0,step=1;')
    else:
        code.append('''unsigned long long warp=(blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x)/32;
if(warp>=nv*stripes)return;unsigned int lane=threadIdx.x%32;
unsigned long long j=warp/stripes,stripe=warp%stripes,offset=stripe*32+lane,step=stripes*32;''')
    code.append(f'''
U shift=mul(vs[j],{config.A}ULL),lower=add(shift,U{{lo0,lo1}}),upper=add(shift,U{{hi0,hi1}});
unsigned long long left=0,right=nu,pop[{groups+1}]={{0}};
while(left<right){{unsigned long long mid=left+(right-left)/2;
 if(!less(lower,mul(us[mid],{config.B}ULL)))left=mid+1;else right=mid;}}
for(unsigned long long i=left+offset;i<nu;i+=step){{U positive=mul(us[i],{config.B}ULL);if(less(upper,positive))break;
if(less(positive,shift)){{atomicAdd(counts+{groups+2},1ULL);continue;}}
U x=sub(positive,shift);++pop[0];''')
    for index,(q,offset) in enumerate(zip(residuals,offsets)):
        memory='frozen' if variant=='constant' else 'allowed'
        lookup=f'{memory}[{offset}ULL+m]' if variant=='byte' else f'(({memory}[{offset}ULL+m/32]>>(m%32))&1U)'
        code.append(f'{{unsigned int m=((x.hi%{q}ULL)*{(1<<64)%q}ULL+x.lo%{q}ULL)%{q}ULL;if(!{lookup})continue;}}')
        if index%4==3 or index+1==len(residuals):code.append(f'++pop[{index//4+1}];')
    code.append(f'''unsigned long long slot=atomicAdd(counts+{groups+1},1ULL);
if(slot<cap){{out[2*slot]=x.lo;out[2*slot+1]=x.hi;}}
}}''')
    if layout=='serial':
        code.append(f'for(unsigned int g=0;g<{groups+1};++g)atomicAdd(counts+g,pop[g]);}}')
    else:
        # Every lane of a live warp reaches this point, even on an empty tail.
        code.append(f'''for(unsigned int g=0;g<{groups+1};++g){{
unsigned long long total=pop[g];
for(unsigned int delta=16;delta;delta>>=1)total+=__shfl_down_sync(0xffffffffU,total,delta);
if(lane==0 && total)atomicAdd(counts+g,total);
}}}}''')
    return '\n'.join(code),table,dtype


class Matcher:
    def __init__(self, runtime: Runtime, config: Config, variant: str = 'byte',
                 layout: str = 'warp', blocks_per_sm: int = 2) -> None:
        self.runtime,self.config,self.variant=runtime,config,variant
        self.cp=runtime.cp
        self.layout,self.blocks_per_sm=layout,blocks_per_sm
        self.sm_count=int(self.cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount'])
        launch_plan(0,0,self.sm_count,layout,blocks_per_sm)
        source,table,dtype=match_source(config,variant,layout)
        self.kernel=runtime.compile(source,'match')
        self.table=self.cp.asarray(np.asarray(table or [0],dtype=np.uint8 if dtype=='unsigned char' else np.uint32))
        residuals=config.residuals();residuals.sort(key=lambda q:(q!=9 and q%3!=1,q))
        self.filter_groups=[residuals[i:i+4] for i in range(0,len(residuals),4)]

    def arrays(self, us: Any, vs: Any, cap: int = 8192) -> dict[str,Any]:
        started=time.perf_counter();ng=len(self.filter_groups)
        if int(us.size)*int(vs.size)>50_000_000:raise OverflowError('bounded pair-opportunity limit exceeded')
        if cap<0:raise ValueError('negative output capacity')
        counts=self.cp.zeros(ng+3,dtype=self.cp.uint64);output=self.cp.empty(max(1,2*cap),dtype=self.cp.uint64)
        blocks,stripes=launch_plan(int(us.size),int(vs.size),self.sm_count,self.layout,self.blocks_per_sm)
        elapsed=0.0
        if us.size and vs.size:
            c=self.config
            elapsed=self.runtime.measured(self.kernel,(blocks,),
              (us,np.uint64(us.size),vs,np.uint64(vs.size),np.uint64(c.lo&MASK),np.uint64(c.lo>>64),
               np.uint64(c.hi&MASK),np.uint64(c.hi>>64),self.table,counts,output,np.uint64(cap),np.uint64(stripes)))
        transfer=time.perf_counter();populations=[int(x) for x in counts.get()]
        number=populations[ng+1]
        if populations[ng+2]:raise ArithmeticError('unsigned underflow flagged')
        if number>cap:raise OverflowError('output overflow; attempt cannot commit')
        limbs=output[:2*number].get().reshape((-1,2))
        values=sorted(int(a)+(int(b)<<64) for a,b in limbs)
        transfer=time.perf_counter()-transfer
        verification=time.perf_counter()
        if len(values)!=len(set(values)):raise ArithmeticError('duplicate canonical output')
        if any(not self.config.lo<n<=self.config.hi or not local3(n,self.config.p) for n in values):
            raise ArithmeticError('GPU/CPU disagreement: attempt quarantined')
        cube_flags=[is_perfect_cube(n) for n in values]
        cubes=[n for n,is_cube in zip(values,cube_flags) if is_cube]
        hits=[n for n,is_cube in zip(values,cube_flags) if not is_cube]
        verification=time.perf_counter()-verification
        return dict(status='PASS',host_verified=True,variant=self.variant,focus_u=int(us.size),focus_v=int(vs.size),
                    layout=self.layout,blocks_per_sm=self.blocks_per_sm,warps_per_row=stripes,
                    launch_blocks=blocks,threads_per_block=256,launched_warps=blocks*8,
                    cartesian_opportunities=int(us.size)*int(vs.size),pairs_visited=populations[0],
                    valid_focused_candidates=populations[0],wide_constructions=populations[0],
                    filter_groups=self.filter_groups,group_survivors=populations[1:ng+1],local_survivors=number,
                    cube_checks=number,genuine_cubes=len(cubes),noncube_reports=len(hits),cpu_confirmed_hits=list(map(str,hits)),
                    cpu_rejected_reports=0,all_values=list(map(str,values)),kernel_seconds=elapsed,
                    transfer_seconds=transfer,verification_seconds=verification,end_to_end_seconds=time.perf_counter()-started)

    def coordinates(self, us: list[int], vs: list[int], cap: int = 8192) -> dict[str,Any]:
        started=time.perf_counter()
        u=self.cp.asarray(np.array(sorted(us),dtype=np.uint64));v=self.cp.asarray(np.array(vs,dtype=np.uint64))
        self.cp.cuda.Stream.null.synchronize();transfer=time.perf_counter()-started
        result=self.arrays(u,v,cap);result['input_transfer_seconds']=transfer
        result['end_to_end_seconds']=time.perf_counter()-started
        return result

    def tile(self, tile: Tile, af: GPUFocus, bf: GPUFocus, cap: int = 1000000, output_cap: int = 8192) -> dict[str,Any]:
        tile.validate(self.config);started=time.perf_counter()
        lower=self.config.u_interval(tile.first)[0];upper=self.config.u_interval(tile.end-1)[1]
        us=af.window(lower,upper,cap);vs=bf.window(tile.first,tile.end,cap)
        prep=time.perf_counter()-started
        result=self.arrays(us,vs,output_cap)
        result['focus_enumeration_seconds']=prep;result['tile']=tile.manifest(self.config)
        result['end_to_end_seconds']=time.perf_counter()-started
        return result
