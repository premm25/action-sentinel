"""SQLite store. Rows hold the full contract as JSON plus a few indexed columns.

Thread-safe via one connection guarded by a lock (adequate for a single local process). Multiple
processes (API + dashboard) can share one file; SQLite's own locking serializes their writes and
WAL mode keeps readers from blocking.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from sentinel.contracts import (
    AuditEvent,
    AuthorizationGrant,
    EvaluationReport,
    ExecutionResult,
    GovernanceDecision,
    ReviewRecord,
    ReviewState,
)

from .base import ConflictError

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, correlation_id TEXT NOT NULL,
  status TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS decisions_event ON decisions(event_id);
CREATE INDEX IF NOT EXISTS decisions_status ON decisions(status);
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, state TEXT NOT NULL,
  created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS reviews_state ON reviews(state);
CREATE TABLE IF NOT EXISTS grants (
  grant_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY, grant_id TEXT NOT NULL, execution_id TEXT NOT NULL, claimed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS executions (
  execution_id TEXT PRIMARY KEY, idempotency_key TEXT, status TEXT NOT NULL,
  started_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, event_type TEXT NOT NULL,
  correlation_id TEXT NOT NULL, decision_id TEXT, timestamp TEXT NOT NULL, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS audit_corr ON audit_events(correlation_id);
CREATE INDEX IF NOT EXISTS audit_dec ON audit_events(decision_id);
CREATE TABLE IF NOT EXISTS evaluations (
  run_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, n_examples INTEGER NOT NULL,
  body TEXT NOT NULL);
"""


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Every statement runs *and is fully consumed* under the lock. Stepping a cursor outside the lock on a
    # shared connection lets another thread's COMMIT fail mid-transaction, which can silently roll back
    # unrelated writes (this caused a double execution under concurrency before it was fixed).
    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    # ------------------------------------------------------------------ decisions
    def save_decision(self, d: GovernanceDecision) -> None:
        try:
            self._write("INSERT INTO decisions VALUES (?,?,?,?,?,?)",
                       (d.decision_id, d.event_id, d.correlation_id, d.status.value, d.created_at.isoformat(),
                        d.model_dump_json()))
        except sqlite3.IntegrityError as exc:
            raise ConflictError("decision for event %s already exists" % d.event_id) from exc

    def update_decision(self, d: GovernanceDecision) -> None:
        n = self._write("UPDATE decisions SET status=?, body=? WHERE decision_id=?",
                         (d.status.value, d.model_dump_json(), d.decision_id))
        if n != 1:
            raise KeyError(d.decision_id)

    def get_decision(self, decision_id: str) -> GovernanceDecision | None:
        row = self._one("SELECT body FROM decisions WHERE decision_id=?", (decision_id,))
        return GovernanceDecision.model_validate_json(row["body"]) if row else None

    def find_decision_by_event(self, event_id: str) -> GovernanceDecision | None:
        row = self._one("SELECT body FROM decisions WHERE event_id=?", (event_id,))
        return GovernanceDecision.model_validate_json(row["body"]) if row else None

    def list_decisions(self, limit: int = 50, offset: int = 0, status: str | None = None) -> list[GovernanceDecision]:
        if status:
            rows = self._all("SELECT body FROM decisions WHERE status=? ORDER BY created_at DESC, rowid DESC "
                              "LIMIT ? OFFSET ?", (status, limit, offset))
        else:
            rows = self._all("SELECT body FROM decisions ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                              (limit, offset))
        return [GovernanceDecision.model_validate_json(r["body"]) for r in rows]

    def count_decisions_by_status(self) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) n FROM decisions GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------------ reviews
    def save_review(self, r: ReviewRecord) -> None:
        self._write("INSERT INTO reviews VALUES (?,?,?,?,?)",
                   (r.review_id, r.decision_id, r.state.value, r.created_at.isoformat(), r.model_dump_json()))

    def update_review(self, r: ReviewRecord, expected_state: ReviewState) -> None:
        """Compare-and-set on state, so two concurrent reviewers cannot both complete a review."""
        n = self._write("UPDATE reviews SET state=?, body=? WHERE review_id=? AND state=?",
                         (r.state.value, r.model_dump_json(), r.review_id, expected_state.value))
        if n != 1:
            raise ConflictError("review %s changed concurrently or does not exist" % r.review_id)

    def get_review(self, review_id: str) -> ReviewRecord | None:
        row = self._one("SELECT body FROM reviews WHERE review_id=?", (review_id,))
        return ReviewRecord.model_validate_json(row["body"]) if row else None

    def list_reviews(self, state: str | None = None, limit: int = 50, offset: int = 0) -> list[ReviewRecord]:
        if state:
            rows = self._all("SELECT body FROM reviews WHERE state=? ORDER BY created_at, rowid LIMIT ? OFFSET ?",
                              (state, limit, offset))
        else:
            rows = self._all("SELECT body FROM reviews ORDER BY created_at, rowid LIMIT ? OFFSET ?",
                              (limit, offset))
        return [ReviewRecord.model_validate_json(r["body"]) for r in rows]

    # ------------------------------------------------------------------ grants / execution
    def save_grant(self, g: AuthorizationGrant) -> None:
        self._write("INSERT INTO grants VALUES (?,?,?)", (g.grant_id, g.decision_id, g.model_dump_json()))

    def update_grant(self, g: AuthorizationGrant) -> None:
        self._write("UPDATE grants SET body=? WHERE grant_id=?", (g.model_dump_json(), g.grant_id))

    def get_grant(self, grant_id: str) -> AuthorizationGrant | None:
        row = self._one("SELECT body FROM grants WHERE grant_id=?", (grant_id,))
        return AuthorizationGrant.model_validate_json(row["body"]) if row else None

    def claim_idempotency_key(self, key: str, grant_id: str, execution_id: str) -> str | None:
        """Atomically claim a key. Returns None on success, or the execution id that already holds it."""
        from sentinel.contracts import utcnow
        with self._lock:
            try:
                self._conn.execute("INSERT INTO idempotency VALUES (?,?,?,?)",
                                   (key, grant_id, execution_id, utcnow().isoformat()))
                return None
            except sqlite3.IntegrityError:
                row = self._conn.execute("SELECT execution_id FROM idempotency WHERE key=?", (key,)).fetchone()
                return row["execution_id"]

    def save_execution(self, e: ExecutionResult) -> None:
        self._write("INSERT OR REPLACE INTO executions VALUES (?,?,?,?,?)",
                   (e.execution_id, e.idempotency_key, e.status, e.started_at.isoformat(), e.model_dump_json()))

    def get_execution(self, execution_id: str) -> ExecutionResult | None:
        row = self._one("SELECT body FROM executions WHERE execution_id=?", (execution_id,))
        return ExecutionResult.model_validate_json(row["body"]) if row else None

    def list_executions(self, limit: int = 50, offset: int = 0) -> list[ExecutionResult]:
        rows = self._all("SELECT body FROM executions ORDER BY started_at DESC, rowid DESC LIMIT ? OFFSET ?",
                          (limit, offset))
        return [ExecutionResult.model_validate_json(r["body"]) for r in rows]

    # ------------------------------------------------------------------ audit
    def append_event(self, e: AuditEvent) -> AuditEvent:
        with self._lock:
            # One transaction: the row and its body (which embeds the assigned seq) appear together or not at all.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "INSERT INTO audit_events (event_id, event_type, correlation_id, decision_id, timestamp, body) "
                    "VALUES (?,?,?,?,?,?)",
                    (e.event_id, e.event_type, e.correlation_id, e.decision_id, e.timestamp.isoformat(), "{}"))
                stored = e.model_copy(update={"seq": cur.lastrowid})
                self._conn.execute("UPDATE audit_events SET body=? WHERE seq=?",
                                   (stored.model_dump_json(), cur.lastrowid))
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
        return stored

    def list_events(self, decision_id: str | None = None, correlation_id: str | None = None,
                    event_type: str | None = None, limit: int = 500, offset: int = 0) -> list[AuditEvent]:
        clauses, params = [], []
        for col, val in (("decision_id", decision_id), ("correlation_id", correlation_id),
                         ("event_type", event_type)):
            if val:
                clauses.append("%s=?" % col)
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._all("SELECT body FROM audit_events%s ORDER BY seq LIMIT ? OFFSET ?" % where,
                          (*params, limit, offset))
        return [AuditEvent.model_validate_json(r["body"]) for r in rows]

    # ------------------------------------------------------------------ evaluation
    def save_evaluation(self, r: EvaluationReport) -> None:
        self._write("INSERT OR REPLACE INTO evaluations VALUES (?,?,?,?,?)",
                   (r.run_id, r.config.name, r.created_at.isoformat(), r.n_examples, r.model_dump_json()))

    def get_evaluation(self, run_id: str) -> EvaluationReport | None:
        row = self._one("SELECT body FROM evaluations WHERE run_id=?", (run_id,))
        return EvaluationReport.model_validate_json(row["body"]) if row else None

    def list_evaluations(self, limit: int = 50) -> list[dict]:
        rows = self._all("SELECT run_id, name, created_at, n_examples FROM evaluations ORDER BY created_at DESC "
                          "LIMIT ?", (limit,))
        return [dict(r) for r in rows]
