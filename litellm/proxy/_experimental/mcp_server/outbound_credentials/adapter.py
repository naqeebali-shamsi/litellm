"""The v1 <-> v2 bridge for the credential resolver.

These edge functions translate v1's request objects into the resolver's typed inputs and map
its typed errors onto the proxy's public exception contract. They import v1 and live outside the
package's public surface so the resolver core (``resolver.py`` / ``types.py``) stays v1-free.
Nothing wires them into ``_create_mcp_client`` yet.

``to_server_spec`` maps only the modes the resolver has gone live for, returning ``None`` for
every other mode so the caller defers to v1 (parity-safe); it grows one branch per migrated mode.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Dict, NoReturn, Optional

from fastapi import HTTPException
from pydantic import SecretStr
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    CredError,
    NoneConfig,
    ServerSpec,
    SharedKey,
    Subject,
)
from litellm.types.mcp import MCPAuth

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


def to_subject(
    user_api_key_auth: Optional["UserAPIKeyAuth"], subject_token: Optional[str]
) -> Subject:
    """Map v1's authenticated principal onto the resolver's Subject.

    tenant_id / subject_id are empty for an unauthenticated caller; the per-user arms must reject
    an empty subject rather than share one credential slot across callers.
    """
    inbound = SecretStr(subject_token) if subject_token else None
    if user_api_key_auth is None:
        return Subject(tenant_id="", subject_id="", inbound_token=inbound)
    return Subject(
        tenant_id=user_api_key_auth.org_id or user_api_key_auth.team_id or "",
        subject_id=user_api_key_auth.user_id or "",
        inbound_token=inbound,
    )


_STATIC_AUTHORIZATION_PREFIX: Dict[MCPAuth, str] = {
    MCPAuth.bearer_token: "Bearer",
    MCPAuth.token: "token",
    MCPAuth.authorization: "",
    MCPAuth.basic: "Basic",
}


def to_server_spec(server: "MCPServer") -> Optional[ServerSpec]:
    """Map a v1 server onto a ServerSpec for a migrated mode, or None to defer to v1.

    Live modes: ``none`` (no upstream credential) and the static-header family (``api_key`` plus
    the Authorization schemes), all shared-key. Every other mode returns None and stays on v1.
    """
    resource = server.url or server.server_id
    auth_type = server.auth_type
    if auth_type is None or auth_type == MCPAuth.none:
        if server.is_oauth_passthrough:
            return None  # passthrough is not migrated yet -> defer to v1
        return ServerSpec(
            server_id=server.server_id, resource=resource, config=NoneConfig()
        )
    if auth_type == MCPAuth.api_key:
        if server.is_byok:
            return None  # per-user BYOK source is not migrated yet -> defer
        return _shared_key_spec(server, resource, "X-API-Key", "")
    prefix = _STATIC_AUTHORIZATION_PREFIX.get(auth_type)
    if prefix is not None:
        # basic carries base64(user:pass); the others send the token verbatim under the scheme.
        token = server.authentication_token
        if not token:
            return None
        value = (
            base64.b64encode(token.encode("utf-8")).decode()
            if auth_type == MCPAuth.basic
            else token
        )
        return _api_key_spec(server.server_id, resource, "Authorization", prefix, value)
    return (
        None  # oauth2 / token_exchange / client_credentials / aws_sigv4 -> defer to v1
    )


def _shared_key_spec(
    server: "MCPServer", resource: str, header_name: str, value_prefix: str
) -> Optional[ServerSpec]:
    token = server.authentication_token
    if not token:
        return None  # no key configured -> defer to v1 (parity-safe)
    return _api_key_spec(server.server_id, resource, header_name, value_prefix, token)


def _api_key_spec(
    server_id: str, resource: str, header_name: str, value_prefix: str, value: str
) -> ServerSpec:
    return ServerSpec(
        server_id=server_id,
        resource=resource,
        config=ApiKeyConfig(
            header_name=header_name,
            value_prefix=value_prefix,
            key_source=SharedKey(value=SecretStr(value)),
        ),
    )


def raise_public(error: CredError) -> NoReturn:
    """Map a resolver CredError onto the proxy's public HTTP contract. The one edge that raises."""
    match error.tag:
        case "unauthorized":
            raise HTTPException(status_code=401, detail=error.summary)
        case "misconfigured":
            raise HTTPException(status_code=500, detail=error.summary)
        case "upstream_unavailable":
            raise HTTPException(status_code=503, detail=error.summary)
        case "unsupported_mode":
            raise HTTPException(status_code=500, detail=error.summary)
        case "precondition_required":
            raise HTTPException(status_code=412, detail=error.summary)
        case "not_implemented":
            raise HTTPException(status_code=501, detail=error.summary)
    assert_never(error.tag)
