"""
Typed configuration settings loaded from config.yaml.
Import pattern: from configs.settings import settings
Access pattern: settings.models.v1.name
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    version: str
    description: str
    max_length: int
    device: str
    quantized: bool = False


@dataclass
class WarmupConfig:
    num_requests: int
    dummy_text: str


@dataclass
class ModelsConfig:
    v1: ModelConfig
    v2: ModelConfig
    warmup: WarmupConfig


@dataclass
class CanaryTrafficSplits:
    canary_5: float
    canary_25: float
    canary_50: float
    full: float


@dataclass
class AutoProgressionConfig:
    enabled: bool
    canary_5_duration_minutes: int
    canary_25_duration_minutes: int
    canary_50_duration_minutes: int


@dataclass
class RollbackConfig:
    error_rate_threshold: float
    latency_p99_multiplier: float
    min_requests_before_check: int


@dataclass
class DeploymentConfig:
    initial_state: str
    canary_traffic_splits: CanaryTrafficSplits
    auto_progression: AutoProgressionConfig
    rollback: RollbackConfig
    state_persistence_path: str
    audit_log_path: str


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    success_threshold: int
    timeout_seconds: int
    call_timeout_seconds: float


@dataclass
class CacheConfig:
    host: str
    port: int
    db: int
    ttl_seconds: int
    max_text_length_for_cache: int
    key_prefix: str


@dataclass
class LatencySLO:
    p50_ms: int
    p95_ms: int
    p99_ms: int


@dataclass
class DisagreementConfig:
    window_size: int
    alert_threshold: float
    min_samples_before_alert: int


@dataclass
class DriftConfig:
    reference_window_size: int
    detection_window_size: int
    run_every_n_requests: int
    text_length_drift_threshold: float
    confidence_drift_threshold: float


@dataclass
class AuditConfig:
    max_entries_in_memory: int


@dataclass
class MonitoringConfig:
    latency_slo: LatencySLO
    disagreement: DisagreementConfig
    drift: DriftConfig
    audit: AuditConfig


@dataclass
class APIConfig:
    host: str
    port: int
    workers: int
    log_level: str
    request_timeout_seconds: float


@dataclass
class GradioConfig:
    host: str
    port: int
    api_base_url: str
    api_base_url_local: str
    theme: str


@dataclass
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    table_audit: str
    table_predictions: str


@dataclass
class Settings:
    models: ModelsConfig
    deployment: DeploymentConfig
    circuit_breaker: CircuitBreakerConfig
    cache: CacheConfig
    monitoring: MonitoringConfig
    api: APIConfig
    gradio: GradioConfig
    postgres: PostgresConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_settings() -> Settings:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        raw: Dict = yaml.safe_load(f)

    m = raw["models"]
    d = raw["deployment"]
    cb = raw["circuit_breaker"]
    c = raw["cache"]
    mon = raw["monitoring"]
    api = raw["api"]
    g = raw["gradio"]
    pg = raw["postgres"]

    # Allow environment variable overrides for Docker / Railway
    redis_host = os.getenv("REDIS_HOST", c["host"])
    pg_host = os.getenv("POSTGRES_HOST", pg["host"])
    pg_password = os.getenv("POSTGRES_PASSWORD", pg["password"])
    gateway_url = os.getenv("GATEWAY_URL", g["api_base_url"])

    return Settings(
        models=ModelsConfig(
            v1=ModelConfig(**m["v1"]),
            v2=ModelConfig(**m["v2"]),
            warmup=WarmupConfig(**m["warmup"]),
        ),
        deployment=DeploymentConfig(
            initial_state=d["initial_state"],
            canary_traffic_splits=CanaryTrafficSplits(**d["canary_traffic_splits"]),
            auto_progression=AutoProgressionConfig(**d["auto_progression"]),
            rollback=RollbackConfig(**d["rollback"]),
            state_persistence_path=d["state_persistence_path"],
            audit_log_path=d["audit_log_path"],
        ),
        circuit_breaker=CircuitBreakerConfig(**cb),
        cache=CacheConfig(
            host=redis_host,
            port=c["port"],
            db=c["db"],
            ttl_seconds=c["ttl_seconds"],
            max_text_length_for_cache=c["max_text_length_for_cache"],
            key_prefix=c["key_prefix"],
        ),
        monitoring=MonitoringConfig(
            latency_slo=LatencySLO(**mon["latency_slo"]),
            disagreement=DisagreementConfig(**mon["disagreement"]),
            drift=DriftConfig(**mon["drift"]),
            audit=AuditConfig(**mon["audit"]),
        ),
        api=APIConfig(**api),
        gradio=GradioConfig(
            host=g["host"],
            port=g["port"],
            api_base_url=gateway_url,
            api_base_url_local=g["api_base_url_local"],
            theme=g["theme"],
        ),
        postgres=PostgresConfig(
            host=pg_host,
            port=pg["port"],
            database=pg["database"],
            user=pg["user"],
            password=pg_password,
            table_audit=pg["table_audit"],
            table_predictions=pg["table_predictions"],
        ),
    )


settings: Settings = _load_settings()
