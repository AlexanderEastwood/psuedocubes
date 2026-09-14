"""Shared local SQLite frontier; ownership survives restarts and never expires."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid
import time

from .accounting import Ledger, digest, encoded
from .focus import Config
from .production import load_production, save_compressed

DEVICES = (
    'GPU-88d1c9ce-f70f-4ee3-05ba-350eb3a726be',
    'GPU-3ad74105-fba6-b76a-9571-a248e4e57758',
    'GPU-b704e2c1-b3cd-fe77-f50c-47b494c7633e',
    'GPU-c807c219-7dbe-155c-0244-9aeea29115f6',
)
Job = tuple[int, str, int, int]


def load_multi(path: Path) -> tuple[dict[str, Any], dict[str, Any], Config]:
    raw = json.loads(path.read_text())
    expected = dict(schema_version=1, authorization='Deploy on the other idle GPUs',
                    base_config='pseudocube_619_production.json', device_uuids=list(DEVICES),
                    ownership='one-unfinished-lease-per-device-no-expiry',
                    coverage='shared-contiguous-frontier-with-out-of-order-commits')
    if raw != expected:
        raise ValueError('unreviewed multi-GPU allocation or scheduling policy')
    base, math = load_production(path.parent / raw['base_config'])
    return raw, base, math


def validate_result(first: int, end: int, result: dict[str, Any]) -> None:
    if result.get('status') != 'PASS' or result.get('host_verified') is not True or result.get('overflow') or result.get('kernel_error'):
        raise ValueError('unverified execution cannot become coverage')
    cursor = first
    for tile in result['subtile_scopes']:
        if int(tile['first']) != cursor or not cursor < int(tile['end']) <= end:
            raise ValueError('gap or overlap in execution subtiles')
        cursor = int(tile['end'])
    if cursor != end:
        raise ValueError('incomplete execution receipt')
    for key in ('valid_focused_candidates', 'genuine_cubes'):
        if int(result[key]) < 0:
            raise ValueError('negative execution count')
    if result.get('receipt_format') == 'compact-batch-v1':
        width = int(result['tile_width'])
        batch = int(result['compute_batch_width'])
        if width <= 0 or batch != 64 * width or len(result['subtile_scopes']) != 1:
            raise ValueError('invalid compact execution geometry')
        if int(result['subtiles_executed']) != (end-first+width-1)//width:
            raise ValueError('compact subtile count mismatch')
        if int(result['compute_batches']) != (end-first+batch-1)//batch:
            raise ValueError('compact batch count mismatch')
        transcript = result['subtile_scopes'][0]['transcript_sha256']
        if len(transcript) != 64 or any(c not in '0123456789abcdef' for c in transcript):
            raise ValueError('invalid execution transcript hash')
        values = result['all_values']
        if len(values) != len(set(values)) or len(values) != int(result['genuine_cubes']) + len(result['cpu_confirmed_hits']):
            raise ValueError('compact output count mismatch')
        if result['subtile_scopes'][0]['output_sha256'] != digest(values):
            raise ValueError('compact output hash mismatch')
    elif 'subtiles_executed' in result or 'receipt_format' in result:
        raise ValueError('unknown receipt format')


def subtile_count(result: dict[str, Any]) -> int:
    return int(result.get('subtiles_executed', len(result['subtile_scopes'])))


class SharedFrontier(Ledger):
    def __init__(self, directory: Path, config: dict[str, Any], build: str,
                 end: int, batch_width: int, owners: tuple[str, ...],
                 baseline: dict[str, Any] | None = None,
                 profile: dict[str, Any] | None = None) -> None:
        super().__init__(directory, config, build)
        self.end, self.batch_width, self.owners = end, batch_width, owners
        self.db.execute('PRAGMA busy_timeout=30000')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS leases(
            first INTEGER PRIMARY KEY, end INTEGER NOT NULL, owner TEXT NOT NULL,
            state TEXT NOT NULL, token TEXT, receipt TEXT, sha TEXT,
            CHECK(first>=0 AND end>first), CHECK(state IN ('pending','running','committed')));
          CREATE UNIQUE INDEX IF NOT EXISTS one_unfinished_per_owner
            ON leases(owner) WHERE state!='committed';
          CREATE INDEX IF NOT EXISTS owner_commits ON leases(owner,first);
          CREATE TABLE IF NOT EXISTS lease_attempts(
            token TEXT PRIMARY KEY, first INTEGER NOT NULL, owner TEXT NOT NULL,
            state TEXT NOT NULL, reason TEXT, build TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS candidates(n TEXT PRIMARY KEY, witness TEXT NOT NULL, first INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS worker_totals(owner TEXT PRIMARY KEY, commits INTEGER NOT NULL,
            candidates TEXT NOT NULL, cubes TEXT NOT NULL, subtiles INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS receipt_blobs(token TEXT PRIMARY KEY, data BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS upgrades(id INTEGER PRIMARY KEY, certificate TEXT NOT NULL, sha TEXT NOT NULL);
        ''')
        with self.transaction():
            initial = self.state()
            active = initial.get('active_build', initial.get('frozen_build', build))
            if active != build:
                raise ValueError('incompatible shared frontier: active_build')
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('frozen_build',?)", (build,))
            fixed = dict(active_build=build, runtime_profile=encoded(profile or {}).decode(), domain_end=str(end),
                         batch_width=str(batch_width), owners=encoded(owners).decode())
            for key, value in fixed.items():
                old = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
                if old and old[0] != value:
                    raise ValueError('incompatible shared frontier: ' + key)
                self.db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)', (key, value))
            existing = self.db.execute("SELECT value FROM meta WHERE key='baseline'").fetchone()
            if existing:
                if baseline is not None and existing[0] != encoded(baseline).decode():
                    raise ValueError('imported coverage changed')
            else:
                if baseline is None:
                    raise ValueError('explicit audited baseline required; refusing to start from zero')
                prefix = int(baseline['cursor'])
                if not 0 <= prefix <= end or baseline['campaign_state'] != 'searching':
                    raise ValueError('invalid baseline or candidate already found')
                self._update(dict(baseline=encoded(baseline).decode(), cursor=str(prefix), allocation_cursor=str(prefix),
                                  campaign_state='searching', **{key: str(baseline[key]) for key in ('commits','candidates','cubes','subtiles')}))
            for owner in owners:
                self.db.execute('INSERT OR IGNORE INTO worker_totals VALUES(?,0,?, ?,0)', (owner, '0', '0'))
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('corrupt shared ledger')
        self.epochs = self._epochs()

    def _epochs(self) -> list[tuple[int, str]]:
        state = self.state()
        epochs = [(int(json.loads(state['baseline'])['cursor']), state['frozen_build'])]
        previous = None
        for row in self.db.execute('SELECT * FROM upgrades ORDER BY id'):
            certificate = json.loads(row['certificate'])
            if digest(certificate) != row['sha'] or certificate['previous_upgrade_sha256'] != previous:
                raise ValueError('upgrade certificate chain changed')
            if certificate['config'] != self.identity or certificate['old_build'] != epochs[-1][1]:
                raise ValueError('upgrade provenance mismatch')
            boundary = int(certificate['boundary'])
            if boundary < epochs[-1][0] or boundary > self.end:
                raise ValueError('invalid upgrade boundary')
            epochs.append((boundary, certificate['new_build']))
            previous = row['sha']
        if epochs[-1][1] != state['active_build']:
            raise ValueError('active build has no upgrade certificate')
        if previous is not None:
            if certificate['new_batch_width'] != state['batch_width'] or encoded(certificate['new_profile']).decode() != state['runtime_profile']:
                raise ValueError('active geometry/profile changed')
        return epochs

    def upgrade(self, build: str, batch_width: int, profile: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        """Offline only: caller must hold every owner/GPU lock. Preserve all old receipts."""
        if not evidence or batch_width <= 0 or not build or build == self.build:
            raise ValueError('qualified upgrade evidence and a new build required')
        audit = self.audit_shared()
        if audit['unfinished_leases'] or self.state()['campaign_state'] != 'searching':
            raise ValueError('upgrade requires a stopped, fully committed frontier')
        with self.transaction():
            state = self.state()
            if state['active_build'] != self.build or state['cursor'] != state['allocation_cursor']:
                raise ValueError('frontier changed during upgrade')
            previous = self.db.execute('SELECT sha FROM upgrades ORDER BY id DESC LIMIT 1').fetchone()
            certificate = dict(config=self.identity, old_build=self.build, new_build=build,
                boundary=state['cursor'], old_batch_width=state['batch_width'], new_batch_width=str(batch_width),
                old_profile=json.loads(state['runtime_profile']), new_profile=profile, before=state,
                previous_upgrade_sha256=previous[0] if previous else None, evidence=evidence,
                unix_time=time.time(), authorization='Deploy to production')
            self.db.execute('INSERT INTO upgrades(certificate,sha) VALUES(?,?)', (encoded(certificate).decode(), digest(certificate)))
            self._update(dict(active_build=build, batch_width=str(batch_width), runtime_profile=encoded(profile).decode()))
        return certificate

    def _active(self) -> None:
        if self.state()['active_build'] != self.build:
            raise ValueError('stale runtime after qualified upgrade')

    def _update(self, values: dict[str, str]) -> None:
        self.db.executemany('INSERT OR REPLACE INTO meta VALUES(?,?)', values.items())

    def state(self) -> dict[str, str]:
        return {r['key']: r['value'] for r in self.db.execute('SELECT * FROM meta')}

    def recover(self, owner: str, stopped_owner_evidence: str) -> int:
        if owner not in self.owners or not stopped_owner_evidence:
            raise ValueError('authorized owner and exclusive-lock evidence required')
        with self.transaction():
            self._active()
            row = self.db.execute("SELECT * FROM leases WHERE owner=? AND state='running'", (owner,)).fetchone()
            if not row:
                return 0
            self.db.execute("UPDATE lease_attempts SET state='abandoned',reason=? WHERE token=?",
                            (stopped_owner_evidence, row['token']))
            self.db.execute("UPDATE leases SET state='pending',token=NULL WHERE first=?", (row['first'],))
            return 1

    def next_batch(self, owner: str) -> Job | None:
        if owner not in self.owners:
            raise ValueError('owner outside authorized allocation')
        with self.transaction():
            self._active()
            state = self.state()
            if state['campaign_state'] != 'searching':
                return None
            row = self.db.execute("SELECT * FROM leases WHERE owner=? AND state!='committed'", (owner,)).fetchone()
            if row:
                if row['state'] != 'pending':
                    raise ValueError('owner still has a running lease; explicit recovery required')
                first, end = row['first'], row['end']
            else:
                first = int(state['allocation_cursor'])
                if first >= self.end:
                    return None
                end = min(first + self.batch_width, self.end)
                self.db.execute('INSERT INTO leases(first,end,owner,state) VALUES(?,?,?,?)', (first, end, owner, 'pending'))
                self._update(dict(allocation_cursor=str(end)))
            token = uuid.uuid4().hex
            self.db.execute("UPDATE leases SET state='running',token=? WHERE first=?", (token, first))
            self.db.execute('INSERT INTO lease_attempts VALUES(?,?,?,?,NULL,?)', (token, first, owner, 'running', self.build))
            return first, token, first, end

    def _lease_owned(self, owner: str, job: Job) -> None:
        self._active()
        first, token, start, end = job
        row = self.db.execute('SELECT * FROM leases WHERE first=?', (first,)).fetchone()
        attempt = self.db.execute('SELECT * FROM lease_attempts WHERE token=?', (token,)).fetchone()
        if not row or row['owner'] != owner or row['state'] != 'running' or row['token'] != token or start != first or row['end'] != end:
            raise ValueError('stale, foreign or changed lease')
        if not attempt or attempt['state'] != 'running' or attempt['build'] != self.build or attempt['owner'] != owner:
            raise ValueError('stale attempt or changed build')

    def fail_lease(self, owner: str, job: Job, reason: str) -> None:
        with self.transaction():
            self._lease_owned(owner, job)
            self.db.execute("UPDATE lease_attempts SET state='failed',reason=? WHERE token=?", (reason, job[1]))
            self.db.execute("UPDATE leases SET state='pending',token=NULL WHERE first=?", (job[0],))

    def complete(self, owner: str, job: Job, result: dict[str, Any]) -> None:
        first, token, _, end = job
        validate_result(first, end, result)
        self._lease_owned(owner, job)
        value = dict(config=self.identity, build=self.build, owner=owner, attempt=token,
                     first=str(first), end=str(end), result=result)
        blob = json.loads(self.state()['runtime_profile']).get('receipt_storage') == 'sqlite-gzip'
        data = gzip.compress(encoded(value), compresslevel=6, mtime=0) if blob else b''
        if blob:
            name, sha = 'sqlite:' + token, hashlib.sha256(data).hexdigest()
        else:
            path = self.directory / 'outputs' / (token + '.json.gz')
            name, sha = str(path.relative_to(self.directory)), save_compressed(path, value)
        with self.transaction():
            self._lease_owned(owner, job)
            if blob:
                self.db.execute('INSERT INTO receipt_blobs VALUES(?,?)', (token, data))
            self.db.execute("UPDATE lease_attempts SET state='committed' WHERE token=?", (token,))
            self.db.execute("UPDATE leases SET state='committed',receipt=?,sha=? WHERE first=?",
                            (name, sha, first))
            for hit in result['cpu_confirmed_hits']:
                self.db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?,?)', (str(hit['n']), encoded(hit).decode(), first))
            state = self.state()
            cursor = int(state['cursor'])
            while True:
                row = self.db.execute("SELECT end FROM leases WHERE first=? AND state='committed'", (cursor,)).fetchone()
                if row is None:
                    break
                cursor = row['end']
            additions = dict(commits=1, candidates=int(result['valid_focused_candidates']), cubes=int(result['genuine_cubes']), subtiles=subtile_count(result))
            phase = 'candidate-found' if result['cpu_confirmed_hits'] or state['campaign_state'] == 'candidate-found' else 'complete' if cursor == self.end else 'searching'
            self._update(dict(cursor=str(cursor), campaign_state=phase, **{k: str(int(state[k]) + v) for k, v in additions.items()}))
            totals = self.db.execute('SELECT * FROM worker_totals WHERE owner=?', (owner,)).fetchone()
            self.db.execute('UPDATE worker_totals SET commits=?, candidates=?, cubes=?, subtiles=? WHERE owner=?',
                            tuple(str(int(totals[k]) + additions[k]) for k in ('commits','candidates','cubes','subtiles')) + (owner,))

    def validate_owner_receipt(self, owner: str) -> None:
        row = self.db.execute("SELECT * FROM leases WHERE owner=? AND state='committed' ORDER BY first DESC LIMIT 1", (owner,)).fetchone()
        if row:
            self._receipt(dict(row))

    def _receipt(self, row: dict[str, Any]) -> dict[str, Any]:
        if row['receipt'].startswith('sqlite:'):
            if row['receipt'] != 'sqlite:' + row['token']:
                raise ValueError('receipt blob token mismatch')
            blob = self.db.execute('SELECT data FROM receipt_blobs WHERE token=?', (row['token'],)).fetchone()
            if blob is None:
                raise ValueError('committed receipt blob missing')
            data = blob[0]
        else:
            data = (self.directory / row['receipt']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha']:
            raise ValueError('committed receipt hash changed')
        value = json.loads(gzip.decompress(data))
        expected_build = self.epochs[0][1]
        for boundary, build in self.epochs[1:]:
            if row['first'] < boundary < row['end']:
                raise ValueError('receipt crosses a build transition')
            if row['first'] >= boundary:
                expected_build = build
        if (value['config'], value['build'], value['owner'], value['attempt'], int(value['first']), int(value['end'])) != (self.identity, expected_build, row['owner'], row['token'], row['first'], row['end']):
            raise ValueError('committed receipt identity mismatch')
        attempt = self.db.execute('SELECT state,build FROM lease_attempts WHERE token=?', (row['token'],)).fetchone()
        if not attempt or tuple(attempt) != ('committed', expected_build):
            raise ValueError('committed attempt mismatch')
        validate_result(row['first'], row['end'], value['result'])
        return value['result']

    def audit_shared(self) -> dict[str, Any]:
        """Run against a frozen SQLite backup while receipt files remain immutable."""
        state = self.state()
        self.epochs = self._epochs()
        baseline = json.loads(state['baseline'])
        if 'certificate_sha256' in baseline:
            certificate_path = self.directory.parent / 'migration.json'
            data = certificate_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != baseline['certificate_sha256']:
                raise ValueError('migration certificate changed')
            certificate = json.loads(data)
            archive = self.directory / baseline['imported_coverage']
            if hashlib.sha256((archive / 'coverage.sqlite3').read_bytes()).hexdigest() != certificate['old_database_sha256']:
                raise ValueError('imported single-worker database changed')
            if certificate['boundary_v'] != baseline['cursor']:
                raise ValueError('migration boundary differs from shared baseline')
            for name, sha in certificate['receipt_sha256'].items():
                if hashlib.sha256((archive / name).read_bytes()).hexdigest() != sha:
                    raise ValueError('imported receipt changed')
        allocated = contiguous = int(baseline['cursor'])
        totals = {k: int(baseline[k]) for k in ('commits','candidates','cubes','subtiles')}
        workers = {owner: 0 for owner in self.owners}
        running: list[dict[str, Any]] = []
        for row in self.db.execute('SELECT * FROM leases ORDER BY first'):
            if row['first'] != allocated or row['end'] > self.end or row['owner'] not in self.owners:
                raise ValueError('overlap, gap or invalid range ownership')
            allocated = row['end']
            if row['state'] == 'committed':
                result = self._receipt(dict(row))
                if row['first'] == contiguous:
                    contiguous = row['end']
                totals['commits'] += 1
                totals['candidates'] += int(result['valid_focused_candidates'])
                totals['cubes'] += int(result['genuine_cubes'])
                totals['subtiles'] += subtile_count(result)
                workers[row['owner']] += 1
            else:
                running.append(dict(first=str(row['first']), end=str(row['end']), owner=row['owner'], state=row['state']))
        if str(allocated) != state['allocation_cursor'] or str(contiguous) != state['cursor'] or any(str(v) != state[k] for k, v in totals.items()):
            raise ValueError('shared frontier or accounting mismatch')
        return dict(status='PASS', imported_prefix=baseline['cursor'], committed_v_end=str(contiguous), allocated_v_end=str(allocated),
                    totals=totals, worker_commits=workers, unfinished_leases=running,
                    no_overlap=True, full_interval_complete=contiguous == self.end, independent_arithmetic_replay=False)
