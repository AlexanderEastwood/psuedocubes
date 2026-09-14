"""Configuration-bound SQLite ownership, durable receipts and persistent budgets.

Local POSIX filesystem only: fsync file, rename, fsync directory, then SQLite
FULL synchronous commit. NFS/object-store durability is not supported.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid


def interoperable(value: Any) -> Any:
    if type(value) is int and abs(value) >= 1 << 53:
        return str(value)
    if isinstance(value, dict):
        return {str(k): interoperable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [interoperable(v) for v in value]
    return value


def encoded(value: Any) -> bytes:
    return (json.dumps(interoperable(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def durable(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(value)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


class Ledger:
    def __init__(self, directory: Path, config: dict[str, Any], build: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.identity, self.build = directory, digest(config), build
        self.db = sqlite3.connect(directory / 'coverage.sqlite3', isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS work(id TEXT PRIMARY KEY,scope TEXT NOT NULL,state TEXT NOT NULL,token TEXT,parent TEXT,receipt TEXT,sha TEXT);
          CREATE TABLE IF NOT EXISTS attempts(token TEXT PRIMARY KEY,work_id TEXT NOT NULL,build TEXT NOT NULL,state TEXT NOT NULL,reason TEXT);
          CREATE TABLE IF NOT EXISTS validations(token TEXT PRIMARY KEY,work_id TEXT NOT NULL,build TEXT NOT NULL,purpose TEXT NOT NULL,state TEXT NOT NULL,receipt TEXT,sha TEXT);
          CREATE TABLE IF NOT EXISTS budget(id TEXT PRIMARY KEY,kind TEXT NOT NULL,reserved REAL NOT NULL,charged REAL,state TEXT NOT NULL);
        ''')
        with self.transaction():
            old = self.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
            if old and old['value'] != self.identity:
                raise ValueError('incompatible checkpoint configuration')
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('config',?)", (self.identity,))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        else:
            self.db.execute('COMMIT')

    def register(self, scope: dict[str, Any], parent: str | None = None) -> str:
        identity = digest({'config': self.identity, 'scope': scope})
        self.db.execute('INSERT OR IGNORE INTO work(id,scope,state,parent) VALUES(?,?,?,?)',
                        (identity, encoded(scope).decode(), 'pending', parent))
        return identity

    def plan(self, scopes: list[dict[str, Any]]) -> list[str]:
        """Bind the expected pilot work list before any execution."""
        if len({digest(scope) for scope in scopes}) != len(scopes):
            raise ValueError('duplicate logical work in expected manifest')
        identity = digest(scopes)
        with self.transaction():
            old = self.db.execute("SELECT value FROM meta WHERE key='plan'").fetchone()
            if old and old['value'] != identity:
                raise ValueError('expected logical work manifest changed')
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('plan',?)", (identity,))
            return [self.register(scope) for scope in scopes]

    def claim(self, work_id: str) -> str:
        token = uuid.uuid4().hex
        with self.transaction():
            row = self.db.execute('SELECT state FROM work WHERE id=?', (work_id,)).fetchone()
            if not row or row['state'] not in ('pending', 'failed'):
                raise ValueError('work already owned, split, or committed')
            self.db.execute("UPDATE work SET state='running',token=? WHERE id=?", (token, work_id))
            self.db.execute('INSERT INTO attempts VALUES(?,?,?,?,NULL)', (token, work_id, self.build, 'running'))
        return token

    def fail(self, work_id: str, token: str, reason: str) -> None:
        with self.transaction():
            self._owned(work_id, token)
            self.db.execute("UPDATE attempts SET state='failed',reason=? WHERE token=?", (reason, token))
            self.db.execute("UPDATE work SET state='failed',token=NULL WHERE id=?", (work_id,))

    def reclaim(self, work_id: str, token: str, stopped_owner_evidence: str) -> None:
        if not stopped_owner_evidence:
            raise ValueError('explicit owner-stop evidence required; no automatic lease expiry')
        self.fail(work_id, token, 'owner stopped: ' + stopped_owner_evidence)

    def _owned(self, work_id: str, token: str) -> None:
        row = self.db.execute('SELECT state,token FROM work WHERE id=?', (work_id,)).fetchone()
        attempt = self.db.execute('SELECT build FROM attempts WHERE token=?', (token,)).fetchone()
        if not row or row['state'] != 'running' or row['token'] != token or not attempt or attempt['build'] != self.build:
            raise ValueError('stale ownership token or build')

    def commit(self, work_id: str, token: str, result: dict[str, Any]) -> None:
        if result.get('status') != 'PASS' or result.get('overflow') or result.get('kernel_error') or not result.get('host_verified'):
            self.fail(work_id, token, 'invalid execution receipt')
            raise ValueError('failed/overflow/unverified attempt cannot commit')
        path = self.directory / 'outputs' / (token + '.json')
        # An orphan durable receipt after a crash is not coverage until commit.
        sha = durable(path, {'config': self.identity, 'build': self.build, 'work_id': work_id, 'attempt': token, 'result': result})
        with self.transaction():
            self._owned(work_id, token)
            self.db.execute("UPDATE attempts SET state='committed' WHERE token=?", (token,))
            self.db.execute("UPDATE work SET state='committed',receipt=?,sha=? WHERE id=?", (str(path.relative_to(self.directory)), sha, work_id))

    def split(self, work_id: str) -> list[str]:
        with self.transaction():
            row = self.db.execute('SELECT state,scope FROM work WHERE id=?', (work_id,)).fetchone()
            if not row or row['state'] not in ('pending', 'failed'):
                raise ValueError('cannot split owned or committed work')
            scope = json.loads(row['scope'])
            first, end = int(scope['first']), int(scope['end'])
            if end - first < 2:
                raise ValueError('unsplittable scope')
            middle = (first + end) // 2
            children = [dict(scope, first=str(first), end=str(middle)), dict(scope, first=str(middle), end=str(end))]
            # Envelopes must be recomputed by the caller from coordinate bounds.
            for child in children:
                child.pop('envelope', None)
            self.db.execute("UPDATE work SET state='split',token=NULL WHERE id=?", (work_id,))
            return [self.register(child, work_id) for child in children]

    def begin_validation(self, work_id: str, purpose: str) -> str:
        token = uuid.uuid4().hex
        with self.transaction():
            row = self.db.execute('SELECT state FROM work WHERE id=?', (work_id,)).fetchone()
            if not row or row['state'] != 'committed':
                raise ValueError('validation replay requires committed primary work')
            self.db.execute('INSERT INTO validations VALUES(?,?,?,?,?,NULL,NULL)',
                            (token, work_id, self.build, purpose, 'running'))
        return token

    def finish_validation(self, token: str, result: dict[str, Any]) -> None:
        row = self.db.execute('SELECT * FROM validations WHERE token=?', (token,)).fetchone()
        if not row or row['state'] != 'running' or row['build'] != self.build:
            raise ValueError('stale validation attempt')
        path = self.directory / 'validations' / (token + '.json')
        sha = durable(path, dict(config=self.identity, build=self.build, attempt=token,
                                 work_id=row['work_id'], purpose=row['purpose'], result=result))
        with self.transaction():
            self.db.execute('UPDATE validations SET state=?,receipt=?,sha=? WHERE token=?',
                            ('verified' if result.get('status') == 'PASS' and result.get('host_verified') else 'failed',
                             str(path.relative_to(self.directory)), sha, token))

    def quarantine(self, work_id: str, reason: str) -> None:
        with self.transaction():
            self.db.execute("UPDATE work SET state='quarantined' WHERE id=?", (work_id,))
            self.db.execute("UPDATE attempts SET state='quarantined',reason=? WHERE work_id=?", (reason, work_id))

    def reserve(self, kind: str, seconds: float, maximum: float) -> str:
        if not 0 < seconds <= maximum:
            raise ValueError('invalid budget reservation')
        token = uuid.uuid4().hex
        with self.transaction():
            used = self.db.execute("SELECT coalesce(sum(CASE WHEN state='held' THEN reserved ELSE charged END),0) FROM budget WHERE kind=?", (kind,)).fetchone()[0]
            if used + seconds > maximum:
                raise RuntimeError('BUDGET_EXHAUSTED')
            self.db.execute('INSERT INTO budget VALUES(?,?,?,NULL,?)', (token, kind, seconds, 'held'))
        return token

    def settle(self, token: str, elapsed: float) -> None:
        if elapsed < 0:
            raise ValueError('negative elapsed budget')
        with self.transaction():
            row = self.db.execute('SELECT state FROM budget WHERE id=?', (token,)).fetchone()
            if not row or row['state'] != 'held':
                raise ValueError('budget already closed')
            self.db.execute("UPDATE budget SET state='closed',charged=? WHERE id=?", (elapsed, token))

    def audit(self) -> dict[str, Any]:
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('corrupt SQLite checkpoint')
        counts: dict[str, int] = {}
        results = []
        for row in self.db.execute('SELECT * FROM work'):
            counts[row['state']] = counts.get(row['state'], 0) + 1
            if row['state'] == 'committed':
                data = (self.directory / row['receipt']).read_bytes()
                if hashlib.sha256(data).hexdigest() != row['sha']:
                    raise ValueError('committed output corrupt')
                value = json.loads(data)
                if value['config'] != self.identity or value['work_id'] != row['id'] or value['attempt'] != row['token']:
                    raise ValueError('receipt identity mismatch')
                attempt = self.db.execute('SELECT build,state FROM attempts WHERE token=?', (row['token'],)).fetchone()
                result = value['result']
                if not attempt or attempt['build'] != value['build'] or attempt['state'] != 'committed' or result.get('status') != 'PASS' or not result.get('host_verified') or result.get('overflow') or result.get('kernel_error'):
                    raise ValueError('invalid committed attempt provenance')
                results.append(value['result'])
        validations = []
        for row in self.db.execute('SELECT * FROM validations'):
            if row['receipt']:
                data = (self.directory / row['receipt']).read_bytes()
                value = json.loads(data)
                if hashlib.sha256(data).hexdigest() != row['sha'] or value['work_id'] != row['work_id'] or value['config'] != self.identity or value['build'] != row['build']:
                    raise ValueError('validation replay receipt corrupt')
            validations.append(dict(row))
        budget = [dict(row) for row in self.db.execute('SELECT * FROM budget')]
        return dict(status='PASS', states=counts, committed_results=results, validations=validations, budget=budget,
                    scope='Only explicitly registered pilot tiles; enclosing numerical interval is not complete.')
