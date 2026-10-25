"""Metadata catalog. Works with PostgreSQL in production and SQLite for local runs."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One pipeline execution, so every clip traces back to its config and seed."""

    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    seed: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict] = mapped_column(JSON)


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(50))
    origin: Mapped[str] = mapped_column(String(500))
    sample_rate: Mapped[int] = mapped_column(Integer)


class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    dim_x: Mapped[float] = mapped_column(Float)
    dim_y: Mapped[float] = mapped_column(Float)
    dim_z: Mapped[float] = mapped_column(Float)
    rt60_target: Mapped[float] = mapped_column(Float)


class Clip(Base):
    """One simulated, augmented multichannel recording plus its labels."""

    __tablename__ = "clips"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    split: Mapped[str] = mapped_column(String(10))  # train / val / test
    audio_path: Mapped[str] = mapped_column(String(500))
    features_path: Mapped[str] = mapped_column(String(500))
    n_channels: Mapped[int] = mapped_column(Integer)
    array_type: Mapped[str] = mapped_column(String(20))
    # Labels a model could learn to predict
    azimuth_deg: Mapped[float] = mapped_column(Float)
    distance_m: Mapped[float] = mapped_column(Float)
    rt60_measured: Mapped[float] = mapped_column(Float)
    # Augmentation parameters
    snr_target_db: Mapped[float] = mapped_column(Float)
    snr_achieved_db: Mapped[float] = mapped_column(Float)
    noise_color: Mapped[str] = mapped_column(String(10))
    gain_db: Mapped[float] = mapped_column(Float)
    limiter_scale: Mapped[float] = mapped_column(Float)
    # Descriptors and QC
    stats: Mapped[dict] = mapped_column(JSON)
    passed_qc: Mapped[bool] = mapped_column(Boolean)
    qc_issues: Mapped[list] = mapped_column(JSON)

    source: Mapped[Source] = relationship()
    room: Mapped[Room] = relationship()


def get_session(db_url: str) -> Session:
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return Session(engine)
