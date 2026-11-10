"""Alembic entry point. Online mode only: every migration runs against a real connection."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from research_assistant.db import sqlalchemy_url

config = context.config
if config.config_file_name and not config.attributes.get("from_code"):
    fileConfig(config.config_file_name)  # CLI only: don't reconfigure logging inside pytest


def _url() -> str:
    url = config.attributes.get("url")  # db.migrate() / tests pass it in
    if url is None:
        from research_assistant.config import get_settings

        url = get_settings().database_url
    return sqlalchemy_url(url)


if context.is_offline_mode():
    raise SystemExit("offline (--sql) mode is not supported; run against a database")

engine = create_engine(_url())
with engine.connect() as connection:
    context.configure(connection=connection, transactional_ddl=True)
    with context.begin_transaction():  # Postgres DDL is transactional: a failed migration leaves nothing behind
        context.run_migrations()
engine.dispose()
