"""Pending approval queue for critical actions that require human consent."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class PendingApproval:
    def __init__(
        self,
        approval_id: str,
        action: str,
        description: str,
        payload: Any = None,
    ) -> None:
        self.id = approval_id
        self.action = action
        self.description = description
        self.payload = payload
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status: str = "pending"  # "pending" | "approved" | "rejected"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "payload": self.payload,
            "created_at": self.created_at,
            "status": self.status,
        }


class ApprovalQueue:
    """In-memory queue of critical-action approval requests."""

    # Actions that always require approval
    CRITICAL_ACTIONS = frozenset(
        {"server_restart", "kv_cache_change", "large_context_increase", "emergency_stop"}
    )

    def __init__(self) -> None:
        self._queue: dict[str, PendingApproval] = {}

    def submit(self, action: str, description: str, payload: Any = None) -> PendingApproval:
        approval = PendingApproval(
            approval_id=str(uuid.uuid4()),
            action=action,
            description=description,
            payload=payload,
        )
        self._queue[approval.id] = approval
        return approval

    def get(self, approval_id: str) -> PendingApproval:
        if approval_id not in self._queue:
            raise KeyError(f"Approval not found: {approval_id!r}")
        return self._queue[approval_id]

    def list_pending(self) -> list[dict]:
        return [a.to_dict() for a in self._queue.values() if a.status == "pending"]

    def approve(self, approval_id: str) -> PendingApproval:
        approval = self.get(approval_id)
        if approval.status != "pending":
            raise ValueError(f"Approval {approval_id!r} is already {approval.status}")
        approval.status = "approved"
        return approval

    def reject(self, approval_id: str) -> PendingApproval:
        approval = self.get(approval_id)
        if approval.status != "pending":
            raise ValueError(f"Approval {approval_id!r} is already {approval.status}")
        approval.status = "rejected"
        return approval

    def is_approved(self, approval_id: str) -> bool:
        return self._queue.get(approval_id, PendingApproval("", "", "")).status == "approved"
