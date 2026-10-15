"""submission.account_id; sequence numbers are per account and trade date

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("submission", sa.Column("account_id", sa.String(length=48), nullable=True))
    op.execute("""
        UPDATE submission s SET account_id = r.account_id
        FROM decision d JOIN run r ON r.run_id = d.run_id
        WHERE d.id = s.decision_id
    """)
    op.alter_column("submission", "account_id", nullable=False)
    op.drop_constraint("uq_submission_trade_date", "submission", type_="unique")
    op.create_unique_constraint("uq_submission_account_date_seq", "submission",
                                ["account_id", "trade_date", "seq"])


def downgrade() -> None:
    op.drop_constraint("uq_submission_account_date_seq", "submission", type_="unique")
    op.create_unique_constraint("uq_submission_trade_date", "submission", ["trade_date", "seq"])
    op.drop_column("submission", "account_id")
