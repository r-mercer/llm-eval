"""Alembic migration environment for LLM evaluation database."""

from __future__ import annotations

import logging
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add project src to path for imports
sys.path.insert(0, "src")

# Import SQLModel and all models
from sqlmodel import SQLModel  # noqa: E402

# Import all models to register them with SQLModel.metadata
from llm_eval.db.models import (  # noqa: E402, F401
    Experiment,
    JudgeRun,
    ModelConfig,
    ModelRating,
    Prompt,
    Result,
    Rubric,
    Task,
)

# Target metadata for autogenerate support
target_metadata = SQLModel.metadata


# =============================================================================
# Alembic Config Object
# =============================================================================

config = context.config

# Configure logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set sqlalchemy.url from config (fallback to default if not set)
# This allows the URL to be overridden via alembic.ini or environment
if not config.get_main_option("sqlalchemy.url"):
    # Fallback: try to get from DatabaseSettings
    try:
        from llm_eval.db.session import DatabaseSettings

        settings = DatabaseSettings()
        config.set_main_option("sqlalchemy.url", settings.database_url)
    except Exception:
        # If import fails, use a default - user must configure manually
        pass


# =============================================================================
# Run Migrations
# =============================================================================


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well.  By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script
    output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# Choose migration mode based on whether URL is available offline
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
