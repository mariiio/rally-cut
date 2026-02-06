"""Database client for evaluation framework."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
from dotenv import dotenv_values

if TYPE_CHECKING:
    from psycopg import Connection


@dataclass
class DbConfig:
    """Database connection configuration."""

    database_url: str

    @classmethod
    def from_env(cls) -> DbConfig:
        """Load from environment, falling back to api/.env.

        Uses python-dotenv for robust .env parsing that handles:
        - Comments (lines starting with #)
        - export prefix
        - Quoted values with escaped characters
        - Multiline values
        """
        url = os.getenv("DATABASE_URL")
        if not url:
            # Try loading from api/.env using python-dotenv
            # __file__ = .../analysis/rallycut/evaluation/db.py
            # parents[3] = .../RallyCut (project root)
            api_env = Path(__file__).parents[3] / "api" / ".env"
            if api_env.exists():
                env_values = dotenv_values(api_env)
                url = env_values.get("DATABASE_URL")
        if not url:
            raise ValueError(
                "DATABASE_URL not set. Either set the environment variable or "
                "ensure api/.env exists with DATABASE_URL defined."
            )
        # Remove Prisma-specific query params that psycopg doesn't understand
        if "?schema=" in url:
            url = url.split("?schema=")[0]
        return cls(database_url=url)


def get_connection() -> Connection[tuple[object, ...]]:
    """Get database connection.

    Returns a connection that should be used as a context manager:

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    config = DbConfig.from_env()
    return psycopg.connect(config.database_url)
