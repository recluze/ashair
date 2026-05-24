from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RAGConfig:
    embedding_model: str = "bge-large-en-v1.5"
    embedding_dim: int = 1024
    chunk_size: int = 512
    top_k: int = 5
    similarity_metric: str = "cosine"
    vector_db_path: str = "data/vector_db"


@dataclass
class SandboxConfig:
    simulation_timestep: float = 1.0
    max_reformulation_attempts: int = 3
    simulation_fidelity_tolerance: float = 0.003
    observation_window_seconds: int = 30


@dataclass
class AgentConfig:
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    inference_endpoint: str = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    max_tokens: int = 2048
    temperature: float = 0.0
    chain_of_thought: bool = True


@dataclass
class ZeroTrustConfig:
    proof_of_intent_secret: str = os.getenv("ZT_SECRET", "change-me-in-production")
    token_expiry_seconds: int = 60
    audit_log_path: str = "data/audit.jsonl"


@dataclass
class SafetyEnvelopeConfig:
    swat: Dict[str, tuple] = field(default_factory=lambda: {
        "FIT101": (0.0, 2.5),
        "LIT101": (0.5, 0.8),
        "MV101": (0, 1),
        "P101": (0, 1),
        "P102": (0, 1),
        "AIT201": (0.0, 1.0),
        "AIT202": (0.0, 1.0),
        "AIT203": (0.0, 1.0),
        "FIT201": (0.0, 2.5),
        "MV201": (0, 1),
        "P201": (0, 1),
        "P203": (0, 1),
        "P205": (0, 1),
        "P206": (0, 1),
        "DPIT301": (0.0, 30.0),
        "FIT301": (0.0, 2.5),
        "LIT301": (0.5, 0.8),
        "MV301": (0, 1),
        "MV302": (0, 1),
        "MV303": (0, 1),
        "MV304": (0, 1),
        "P301": (0, 1),
        "P302": (0, 1),
        "AIT401": (0.0, 0.05),
        "AIT402": (0.0, 0.05),
    })
    hai: Dict[str, tuple] = field(default_factory=lambda: {
        "boiler_steam_pressure": (0.0, 10.0),
        "boiler_temperature": (200.0, 900.0),
        "turbine_speed": (0.0, 3600.0),
        "turbine_output_power": (0.0, 500.0),
        "water_level": (0.2, 0.9),
        "feed_water_flow": (0.0, 100.0),
        "condenser_pressure": (0.0, 5.0),
    })
    ics_nad: Dict[str, tuple] = field(default_factory=lambda: {
        "plc_response_time_ms": (0.0, 500.0),
        "network_throughput_mbps": (0.0, 1000.0),
        "packet_loss_rate": (0.0, 0.05),
    })
    psv_limit_pct: float = 2.0


@dataclass
class EvaluationConfig:
    results_path: str = "data/results"
    mttr_timeout_seconds: float = 120.0
    load_restoration_window_seconds: int = 60
    zero_day_holdout_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class ASHAIRConfig:
    rag: RAGConfig = field(default_factory=RAGConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    zero_trust: ZeroTrustConfig = field(default_factory=ZeroTrustConfig)
    safety_envelope: SafetyEnvelopeConfig = field(default_factory=SafetyEnvelopeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    modbus_host: str = os.getenv("MODBUS_HOST", "127.0.0.1")
    modbus_port: int = int(os.getenv("MODBUS_PORT", "502"))
    firewall_api_url: str = os.getenv("FIREWALL_API_URL", "http://localhost:9000")
    container_api_url: str = os.getenv("CONTAINER_API_URL", "http://localhost:2376")
    syslog_host: str = os.getenv("SYSLOG_HOST", "0.0.0.0")
    syslog_port: int = int(os.getenv("SYSLOG_PORT", "514"))
    pcap_interface: str = os.getenv("PCAP_INTERFACE", "eth0")
    opc_ua_endpoint: str = os.getenv("OPC_UA_ENDPOINT", "opc.tcp://localhost:4840")


DEFAULT_CONFIG = ASHAIRConfig()
