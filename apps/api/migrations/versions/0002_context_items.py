"""context_items: the per-run context store (stored tool results + rebuttals, addressable by ref).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-15 20:14:27.805216
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('context_items',
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('ref', sa.String(length=16), nullable=False),
    sa.Column('node_id', sa.UUID(), nullable=False),
    sa.Column('attempt_id', sa.String(length=32), nullable=False),
    sa.Column('node_version_id', sa.UUID(), nullable=True),
    sa.Column('agent', sa.String(length=32), nullable=False),
    sa.Column('parent_ref', sa.String(length=16), nullable=True),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('tool', sa.String(length=64), nullable=False),
    sa.Column('args', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('ok', sa.Boolean(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('chunk_ids', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.Column('tokens_est', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], name='fk_context_items_node_id', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_version_id'], ['node_versions.id'], name='fk_context_items_node_version_id', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], name='fk_context_items_run_id', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('run_id', 'ref', name='pk_context_items')
    )
    op.create_index('ix_context_items_run_version', 'context_items', ['run_id', 'node_version_id'], unique=False)
    op.create_index('ix_context_items_text_fts', 'context_items', [sa.literal_column("to_tsvector('english', text)")], unique=False, postgresql_using='gin')


def downgrade() -> None:
    op.drop_index('ix_context_items_text_fts', table_name='context_items', postgresql_using='gin')
    op.drop_index('ix_context_items_run_version', table_name='context_items')
    op.drop_table('context_items')
