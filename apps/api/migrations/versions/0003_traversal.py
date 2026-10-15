"""traversal: node_versions.output_hash (early cutoff) and steer_messages.source (agent upstream requests).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-15 22:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("node_versions", sa.Column("output_hash", sa.String(length=64), nullable=True))
    op.add_column(
        "steer_messages",
        sa.Column("source", sa.String(length=32), server_default="user", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("steer_messages", "source")
    op.drop_column("node_versions", "output_hash")
