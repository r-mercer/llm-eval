"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel import Session, SQLModel, create_engine

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


# =============================================================================
# Database Configuration
# =============================================================================


class DatabaseSettings(BaseSettings):
    """Database configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "localhost"
    port: int = 5432
    name: str = "llm_eval"
    user: str = "postgres"
    password: str = ""

    # Full URL takes precedence over individual settings
    url: str = ""

    @property
    def database_url(self) -> str:
        """Build PostgreSQL URL from individual settings."""
        if self.url:
            return self.url
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.name}"


# =============================================================================
# Engine Management
# =============================================================================


def create_db_engine(database_url: str) -> "Engine":
    """Create database engine with appropriate settings."""
    # Validate URL format early - fail fast with clear error
    if not database_url:
        raise ValueError("Database URL cannot be empty")

    if not database_url.startswith("postgresql"):
        raise ValueError(f"Unsupported database dialect: {database_url}")

    engine = create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    return engine


# Global engine instance
_engine = None


def get_engine() -> "Engine":
    """Get or create the global database engine."""
    global _engine

    if _engine is None:
        settings = DatabaseSettings()
        _engine = create_db_engine(settings.database_url)

    return _engine


# =============================================================================
# Session Management
# =============================================================================


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a database session with automatic cleanup.

    Usage:
        with get_session() as session:
            session.add(new_record)
            session.commit()
    """
    engine = get_engine()
    session = Session(engine)

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# =============================================================================
# Database Initialization
# =============================================================================


def init_db() -> None:
    """Create all database tables."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
