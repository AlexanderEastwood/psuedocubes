"""Frozen production configuration and a single-owner, contiguous CRT frontier."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .accounting import Ledger,encoded,digest
from .focus import Config,A619,B619


def load_production(path: Path) -> tuple[dict[str,Any],Config]:
    raw=json.loads(path.read_text())
    if raw.get('mode')!='production' or raw.get('schema_version')!=1:
        raise ValueError('explicit production configuration required')
    if raw.get('authorization',{}).get('user_request')!='Wrap it up and deploy to prod':
        raise ValueError('missing recorded production authorization')
    if raw.get('device_uuid')!='GPU-88d1c9ce-f70f-4ee3-05ba-350eb3a726be':
        raise ValueError('production allocation differs from the authorized Quad device')
    if raw.get('p')!=619 or raw.get('lo_exclusive')!=str(10**27) or raw.get('hi_inclusive')!=str(3*10**27):
        raise ValueError('production interval differs from the reviewed interval')
    if raw.get('A')!=str(A619) or raw.get('B')!=str(B619):raise ValueError('focus changed')
    if raw.get('layout')!='warp' or raw.get('blocks_per_sm')!=2 or raw.get('variant')!='byte':
        raise ValueError('untested production matcher')
    if raw.get('tile_width')!=100000000 or raw.get('tiles_per_commit')!=64:
        raise ValueError('unexpected bounded work geometry')
    if raw.get('verification_policy')!='candidate-validity-now-independent-coverage-before-minimum':
        raise ValueError('unknown verification policy')
    if raw.get('stop_on_candidate') is not True or raw.get('allow_range_extension') is not False:
        raise ValueError('production stop/range policy changed')
    if raw.get('max_device_bytes')!=8*1024**3 or raw.get('max_device_fraction')!=0.5 or raw.get('max_host_bytes')!=2*1024**3:
        raise ValueError('memory allocation changed')
    if raw.get('max_state_bytes')!=8*1024**3 or raw.get('min_free_disk_bytes')!=20*1024**3:
        raise ValueError('storage guard changed')
    return raw,Config(619,10**27,3*10**27,A619,B619)


def save_compressed(path: Path, value: dict[str,Any]) -> str:
    data=gzip.compress(encoded(value),mtime=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(data).hexdigest()


class Frontier(Ledger):
    def __init__(self, directory: Path, config: dict[str,Any], build: str, end: int, batch_width: int) -> None:
        super().__init__(directory,config,build)
        self.end,self.batch_width=end,batch_width
        with self.transaction():
            for key,value in dict(frozen_build=build,domain_end=str(end),batch_width=str(batch_width)).items():
                old=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
                if old and old[0]!=value:raise ValueError('incompatible production frontier: '+key)
                self.db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)',(key,value))
            for key,value in dict(cursor='0',campaign_state='searching',commits='0',candidates='0',cubes='0',subtiles='0').items():
                self.db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)',(key,value))
        if self.db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('corrupt production ledger')

    def state(self) -> dict[str,str]:
        return {r['key']:r['value'] for r in self.db.execute('SELECT * FROM meta')}

    def recover_stopped_owner(self, evidence: str) -> int:
        if not evidence:raise ValueError('stopped-owner evidence required')
        rows=list(self.db.execute("SELECT id,token FROM work WHERE state='running'"))
        for row in rows:self.reclaim(row['id'],row['token'],evidence)
        return len(rows)

    def next_batch(self) -> tuple[str,str,int,int] | None:
        state=self.state();first=int(state['cursor'])
        if state['campaign_state']!='searching' or first>=self.end:return None
        end=min(self.end,first+self.batch_width)
        work=self.register(dict(kind='production-v-window',first=str(first),end=str(end)))
        return work,self.claim(work),first,end

    def complete(self, work: str, token: str, first: int, end: int, result: dict[str,Any]) -> None:
        if result.get('status')!='PASS' or not result.get('host_verified') or result.get('overflow') or result.get('kernel_error'):
            self.fail(work,token,'invalid production receipt');raise ValueError('invalid production output')
        tiles=result['subtile_scopes'];cursor=first
        for tile in tiles:
            if int(tile['first'])!=cursor or not cursor<int(tile['end'])<=end:
                raise ValueError('gap or overlap in production subtiles')
            cursor=int(tile['end'])
        if cursor!=end:raise ValueError('incomplete production batch')
        path=self.directory/'outputs'/(token+'.json.gz')
        sha=save_compressed(path,dict(config=self.identity,build=self.build,work_id=work,attempt=token,
                                     first=str(first),end=str(end),result=result))
        with self.transaction():
            self._owned(work,token)
            state=self.state()
            scope=json.loads(self.db.execute('SELECT scope FROM work WHERE id=?',(work,)).fetchone()[0])
            if int(state['cursor'])!=first or int(scope['first'])!=first or int(scope['end'])!=end:
                raise ValueError('frontier moved or scope changed')
            self.db.execute("UPDATE attempts SET state='committed' WHERE token=?",(token,))
            self.db.execute("UPDATE work SET state='committed',receipt=?,sha=? WHERE id=?",(str(path.relative_to(self.directory)),sha,work))
            updates=dict(cursor=str(end),commits=str(int(state['commits'])+1),
                         candidates=str(int(state['candidates'])+int(result['valid_focused_candidates'])),
                         cubes=str(int(state['cubes'])+int(result['genuine_cubes'])),
                         subtiles=str(int(state['subtiles'])+len(tiles)),last_receipt=str(path.relative_to(self.directory)),last_sha=sha,
                         campaign_state='candidate-found' if result['cpu_confirmed_hits'] else 'complete' if end==self.end else 'searching')
            for key,value in updates.items():self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,value))

    def validate_last_receipt(self) -> None:
        state=self.state()
        if state.get('last_receipt'):
            data=(self.directory/state['last_receipt']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=state['last_sha']:raise ValueError('last production receipt corrupt')
            row=json.loads(gzip.decompress(data))
            if row['config']!=self.identity or row['build']!=self.build or row['end']!=state['cursor']:
                raise ValueError('last receipt/frontier mismatch')

    def audit_production(self) -> dict[str,Any]:
        cursor=0;commits=0
        # Single sequential owner commits in row insertion order, independent of UUID.
        for row in self.db.execute("SELECT * FROM work WHERE state='committed' ORDER BY rowid"):
            data=(self.directory/row['receipt']).read_bytes();value=json.loads(gzip.decompress(data))
            if hashlib.sha256(data).hexdigest()!=row['sha'] or value['config']!=self.identity or value['build']!=self.build:
                raise ValueError('corrupt committed production evidence')
            if int(value['first'])!=cursor or value['attempt']!=row['token']:raise ValueError('production coverage gap/overlap')
            cursor=int(value['end']);commits+=1
        if cursor!=int(self.state()['cursor']):raise ValueError('frontier does not match committed receipts')
        return dict(status='PASS',commits=commits,committed_v_end=str(cursor),domain_end=str(self.end),
                    full_interval_complete=cursor==self.end,independent_arithmetic_replay=False)
