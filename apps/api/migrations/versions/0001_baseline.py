"""baseline: the schema as deployed before Alembic (create_all + the old hand-written ALTERs).

Existing databases are stamped at this revision by livegraph.db.migrate (no DDL runs); fresh ones get it
here. Postgres default PK/FK names, uq_/ix_ names as in the models, Vector(1536) hardcoded on purpose.

Revision ID: 0001
Revises: 
Create Date: 2026-10-15 20:13:17.352053
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('documents',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=True),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('source_url', sa.Text(), nullable=True),
    sa.Column('origin', sa.String(length=16), server_default='upload', nullable=False),
    sa.Column('trust', sa.String(length=16), server_default='normal', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_run_id'), 'documents', ['run_id'], unique=False)
    op.create_index('uq_documents_run_source_url', 'documents', ['run_id', 'source_url'], unique=True, postgresql_where=sa.text('source_url IS NOT NULL'))
    op.create_table('runs',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('goal', sa.Text(), nullable=False),
    sa.Column('workflow', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('web_cache',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('query_norm', sa.Text(), nullable=False),
    sa.Column('doc_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'query_norm', name='uq_web_cache_run_query')
    )
    op.create_index(op.f('ix_web_cache_run_id'), 'web_cache', ['run_id'], unique=False)
    op.create_table('chunks',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=False),
    sa.Column('idx', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('embedding', Vector(1536), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chunks_document_id'), 'chunks', ['document_id'], unique=False)
    op.create_index('ix_chunks_embedding_hnsw', 'chunks', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_table('events',
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('seq', sa.BigInteger(), nullable=False),
    sa.Column('type', sa.String(length=32), nullable=False),
    sa.Column('node_id', sa.UUID(), nullable=True),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('run_id', 'seq', name='pk_events')
    )
    op.create_table('nodes',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('key', sa.String(length=64), nullable=False),
    sa.Column('type', sa.String(length=32), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('position', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_by', sa.String(length=16), nullable=False),
    sa.Column('current_version_id', sa.UUID(), nullable=True),
    sa.Column('removed', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'key', name='uq_nodes_run_key')
    )
    op.create_index(op.f('ix_nodes_run_id'), 'nodes', ['run_id'], unique=False)
    op.create_table('claims',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('key', sa.String(length=32), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('source_node_id', sa.UUID(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_node_id'], ['nodes.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('run_id', 'key', name='uq_claims_run_key')
    )
    op.create_index(op.f('ix_claims_run_id'), 'claims', ['run_id'], unique=False)
    op.create_table('edges',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('run_id', sa.UUID(), nullable=False),
    sa.Column('source_node_id', sa.UUID(), nullable=False),
    sa.Column('target_node_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['source_node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['target_node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_node_id', 'target_node_id', name='uq_edges_pair')
    )
    op.create_index(op.f('ix_edges_run_id'), 'edges', ['run_id'], unique=False)
    op.create_table('node_versions',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('node_id', sa.UUID(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('input_hash', sa.String(length=64), nullable=False),
    sa.Column('output', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('output_text', sa.Text(), nullable=False),
    sa.Column('reason', sa.String(length=16), nullable=False),
    sa.Column('steer_text', sa.Text(), nullable=True),
    sa.Column('model', sa.String(length=128), nullable=True),
    sa.Column('tokens_in', sa.Integer(), nullable=True),
    sa.Column('tokens_out', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('trace', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
    sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('node_id', 'version', name='uq_node_versions_node_version')
    )
    op.create_index(op.f('ix_node_versions_input_hash'), 'node_versions', ['input_hash'], unique=False)
    op.create_index(op.f('ix_node_versions_node_id'), 'node_versions', ['node_id'], unique=False)
    op.create_table('steer_messages',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('node_id', sa.UUID(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('consumed_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['node_id'], ['nodes.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_steer_messages_node_id'), 'steer_messages', ['node_id'], unique=False)
    op.create_table('claim_history',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('claim_id', sa.UUID(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('node_version_id', sa.UUID(), nullable=True),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('ts', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_version_id'], ['node_versions.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_claim_history_claim_id'), 'claim_history', ['claim_id'], unique=False)
    op.create_table('retrievals',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('node_version_id', sa.UUID(), nullable=False),
    sa.Column('chunk_id', sa.UUID(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('rank', sa.Integer(), nullable=False),
    sa.Column('cited', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('via', sa.String(length=8), nullable=True),
    sa.ForeignKeyConstraint(['chunk_id'], ['chunks.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['node_version_id'], ['node_versions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_retrievals_chunk_id'), 'retrievals', ['chunk_id'], unique=False)
    op.create_index(op.f('ix_retrievals_node_version_id'), 'retrievals', ['node_version_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_retrievals_node_version_id'), table_name='retrievals')
    op.drop_index(op.f('ix_retrievals_chunk_id'), table_name='retrievals')
    op.drop_table('retrievals')
    op.drop_index(op.f('ix_claim_history_claim_id'), table_name='claim_history')
    op.drop_table('claim_history')
    op.drop_index(op.f('ix_steer_messages_node_id'), table_name='steer_messages')
    op.drop_table('steer_messages')
    op.drop_index(op.f('ix_node_versions_node_id'), table_name='node_versions')
    op.drop_index(op.f('ix_node_versions_input_hash'), table_name='node_versions')
    op.drop_table('node_versions')
    op.drop_index(op.f('ix_edges_run_id'), table_name='edges')
    op.drop_table('edges')
    op.drop_index(op.f('ix_claims_run_id'), table_name='claims')
    op.drop_table('claims')
    op.drop_index(op.f('ix_nodes_run_id'), table_name='nodes')
    op.drop_table('nodes')
    op.drop_table('events')
    op.drop_index('ix_chunks_embedding_hnsw', table_name='chunks', postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_index(op.f('ix_chunks_document_id'), table_name='chunks')
    op.drop_table('chunks')
    op.drop_index(op.f('ix_web_cache_run_id'), table_name='web_cache')
    op.drop_table('web_cache')
    op.drop_table('runs')
    op.drop_index('uq_documents_run_source_url', table_name='documents', postgresql_where=sa.text('source_url IS NOT NULL'))
    op.drop_index(op.f('ix_documents_run_id'), table_name='documents')
    op.drop_table('documents')
