"""Authorization grants: issued only by governance (auto-approve) or a completed human review.

A token is `<grant_id>.<hmac>`. The HMAC binds the grant id, action hash, policy version and expiry,
so a token cannot be edited to point at a different action. The grant row in the store remains
the source of truth (revocation and expiry are checked there too).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from datetime import datetime, timedelta

from action_sentinel.contracts import ActionProposal, AuthorizationGrant, IssuedAuthorization, utcnow


class AuthorizationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Authorizer:
    def __init__(self, store, signing_key: bytes, ttl_seconds: int = 900,
                 clock: Callable[[], datetime] = utcnow):
        if len(signing_key) < 16:
            raise ValueError("signing key must be at least 16 bytes")
        self.store = store
        self._key = signing_key
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    def _sign(self, g: AuthorizationGrant) -> str:
        msg = "|".join([g.grant_id, g.action_hash, g.policy_version, g.expires_at.isoformat(), g.idempotency_key])
        return hmac.new(self._key, msg.encode("utf-8"), hashlib.sha256).hexdigest()

    def issue(self, *, decision_id: str, correlation_id: str, action: ActionProposal, policy_version: str,
              issued_by: str, review_id: str | None = None, approvals: list[str] | None = None) -> IssuedAuthorization:
        now = self.clock()
        g = AuthorizationGrant(
            decision_id=decision_id, review_id=review_id, correlation_id=correlation_id,
            action_hash=action.action_hash(), idempotency_key=action.idempotency_key,
            policy_version=policy_version, issued_at=now, expires_at=now + timedelta(seconds=self.ttl_seconds),
            issued_by=issued_by, approvals=approvals or [])
        self.store.save_grant(g)
        return IssuedAuthorization(grant=g, token="%s.%s" % (g.grant_id, self._sign(g)))

    def token_for(self, grant: AuthorizationGrant) -> str:
        """Re-derive the token for a stored grant (for local tools that hold the signing key)."""
        return "%s.%s" % (grant.grant_id, self._sign(grant))

    def verify(self, token: str) -> AuthorizationGrant:
        """Signature, existence, revocation and expiry. Action matching is the executor's job."""
        if not isinstance(token, str) or token.count(".") != 1:
            raise AuthorizationError("INVALID_TOKEN", "malformed authorization token")
        grant_id, sig = token.split(".")
        g = self.store.get_grant(grant_id)
        if g is None:
            raise AuthorizationError("UNKNOWN_GRANT", "no such authorization")
        if not hmac.compare_digest(sig, self._sign(g)):
            raise AuthorizationError("INVALID_TOKEN", "authorization signature does not verify")
        if g.revoked:
            raise AuthorizationError("AUTHORIZATION_REVOKED", "authorization was revoked: %s" % g.revoked_reason)
        if self.clock() >= g.expires_at:
            raise AuthorizationError("AUTHORIZATION_EXPIRED", "authorization expired at %s" % g.expires_at.isoformat())
        return g

    def revoke(self, grant_id: str, reason: str) -> AuthorizationGrant:
        g = self.store.get_grant(grant_id)
        if g is None:
            raise AuthorizationError("UNKNOWN_GRANT", "no such authorization")
        if not g.revoked:
            g = g.model_copy(update={"revoked": True, "revoked_at": self.clock(), "revoked_reason": reason})
            self.store.update_grant(g)
        return g
