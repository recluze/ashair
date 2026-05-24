#!/usr/bin/env python3
"""
ASHAIR Live Runtime — deploys the full pipeline against a live or emulated CPS.
Usage:
    python main.py --domain swat --modbus-host 192.168.1.100 --poll-interval 1.0
    python main.py --domain hai --dry-run
"""
from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
import time
from queue import Queue, Empty
from threading import Event
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ASHAIRConfig, DEFAULT_CONFIG
from pipeline.orchestrator import ASHAIROrchestrator
from pipeline.ingestion import (
    ModbusTCPPoller, SyslogReceiver, TelemetryNormalizer, RawTelemetryRecord
)
from pipeline.normalizer import TelemetryNarrativeGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ashair.main")

SWAT_REGISTER_MAP = {
    "LIT101": {"address": 1, "unit_id": 1, "scale": 0.001},
    "FIT101": {"address": 2, "unit_id": 1, "scale": 0.01},
    "MV101": {"address": 3, "unit_id": 1, "scale": 1},
    "P101": {"address": 4, "unit_id": 1, "scale": 1},
    "LIT301": {"address": 5, "unit_id": 1, "scale": 0.001},
    "FIT301": {"address": 6, "unit_id": 1, "scale": 0.01},
}

HAI_REGISTER_MAP = {
    "boiler_temperature": {"address": 100, "unit_id": 1, "scale": 0.1},
    "boiler_steam_pressure": {"address": 101, "unit_id": 1, "scale": 0.01},
    "turbine_speed": {"address": 102, "unit_id": 1, "scale": 1.0},
    "water_level": {"address": 103, "unit_id": 1, "scale": 0.001},
}

DOMAIN_REGISTER_MAPS = {"swat": SWAT_REGISTER_MAP, "hai": HAI_REGISTER_MAP}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASHAIR Live CPS Incident Response Runtime")
    parser.add_argument("--domain", type=str, choices=["swat", "hai", "ics_nad"], default="swat")
    parser.add_argument("--modbus-host", type=str, default="127.0.0.1")
    parser.add_argument("--modbus-port", type=int, default=502)
    parser.add_argument("--syslog-host", type=str, default="0.0.0.0")
    parser.add_argument("--syslog-port", type=int, default=514)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--vllm-endpoint", type=str, default=None)
    parser.add_argument("--vector-db-path", type=str, default="data/vector_db")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic telemetry instead of live PLC")
    parser.add_argument("--max-incidents", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ASHAIRConfig:
    config = ASHAIRConfig()
    config.modbus_host = args.modbus_host
    config.modbus_port = args.modbus_port
    config.syslog_host = args.syslog_host
    config.syslog_port = args.syslog_port
    config.rag.vector_db_path = args.vector_db_path
    if args.vllm_endpoint:
        config.agents.inference_endpoint = args.vllm_endpoint
    return config


def generate_synthetic_telemetry(domain: str, index: int) -> RawTelemetryRecord:
    import random
    from datetime import datetime, timezone
    from pipeline.ingestion import RawTelemetryRecord
    timestamp = datetime.now(timezone.utc)
    if domain == "swat":
        payload = {
            "LIT101": 0.65 + random.gauss(0, 0.005),
            "FIT101": 1.2 + random.gauss(0, 0.02),
            "MV101": 1,
            "P101": 1,
            "LIT301": 0.65 + random.gauss(0, 0.005),
            "FIT301": 1.2 + random.gauss(0, 0.02),
            "Normal/Attack": "Normal",
        }
        if index == 50:
            payload["LIT101"] = 0.148
            payload["Normal/Attack"] = "Attack"
    elif domain == "hai":
        payload = {
            "boiler_temperature": 550.0 + random.gauss(0, 2.0),
            "boiler_steam_pressure": 5.0 + random.gauss(0, 0.05),
            "turbine_speed": 1800.0 + random.gauss(0, 5.0),
            "water_level": 0.55 + random.gauss(0, 0.01),
            "attack": 0,
        }
        if index == 50:
            payload["boiler_temperature"] = 950.0
            payload["attack"] = 1
    else:
        payload = {
            "plc_response_time_ms": 50.0,
            "network_throughput_mbps": 100.0,
            "label": "Benign",
        }
    return RawTelemetryRecord(
        source_protocol="synthetic",
        timestamp=timestamp,
        raw_payload=payload,
        channel=domain,
    )


def build_synthetic_twin_state(domain: str) -> dict:
    if domain == "swat":
        return {"LIT101": 0.65, "FIT101": 1.2, "LIT301": 0.65, "FIT301": 1.2}
    elif domain == "hai":
        return {"boiler_temperature": 550.0, "boiler_steam_pressure": 5.0,
                "turbine_speed": 1800.0, "water_level": 0.55}
    return {"plc_response_time_ms": 50.0, "network_throughput_mbps": 100.0}


def run_live_mode(
    orchestrator: ASHAIROrchestrator,
    domain: str,
    config: ASHAIRConfig,
    poll_interval: float,
    stop_event: Event,
    max_incidents: Optional[int],
) -> None:
    telemetry_queue: Queue = Queue(maxsize=1000)
    normalizer = TelemetryNormalizer()
    register_map = DOMAIN_REGISTER_MAPS.get(domain, SWAT_REGISTER_MAP)
    poller = ModbusTCPPoller(
        host=config.modbus_host,
        port=config.modbus_port,
        register_map=register_map,
        poll_interval_s=poll_interval,
        output_queue=telemetry_queue,
    )
    syslog_server = SyslogReceiver(
        host=config.syslog_host,
        port=config.syslog_port,
        output_queue=telemetry_queue,
    )
    poller.start()
    syslog_server.start_background()
    logger.info("ASHAIR live mode started: domain=%s modbus=%s:%d syslog=%s:%d",
                domain, config.modbus_host, config.modbus_port,
                config.syslog_host, config.syslog_port)
    incident_count = 0
    while not stop_event.is_set():
        try:
            raw = telemetry_queue.get(timeout=1.0)
            norm = normalizer.normalize(raw)
            twin_state = build_synthetic_twin_state(domain)
            result = orchestrator.process_telemetry(record=norm, twin_state=twin_state)
            if result.detected:
                incident_count += 1
                logger.info(
                    "Incident %d resolved: MTTR=%.2fs PSV=%.2f%% outcome=%s",
                    incident_count, result.mttr_s or 0.0,
                    result.predicted_psv_pct or 0.0, result.outcome,
                )
                if max_incidents and incident_count >= max_incidents:
                    logger.info("Max incidents reached (%d); stopping", max_incidents)
                    stop_event.set()
        except Empty:
            continue
        except Exception as exc:
            logger.error("Error processing telemetry: %s", exc, exc_info=True)
    poller.stop()


def run_dry_run_mode(
    orchestrator: ASHAIROrchestrator,
    domain: str,
    max_steps: int,
    stop_event: Event,
) -> None:
    normalizer = TelemetryNormalizer()
    logger.info("ASHAIR dry-run mode started: domain=%s max_steps=%d", domain, max_steps)
    for i in range(max_steps):
        if stop_event.is_set():
            break
        raw = generate_synthetic_telemetry(domain, i)
        norm = normalizer.normalize(raw)
        twin_state = build_synthetic_twin_state(domain)
        result = orchestrator.process_telemetry(record=norm, twin_state=twin_state)
        if result.detected:
            logger.info(
                "Step %d: Incident detected MTTR=%.2fs outcome=%s",
                i, result.mttr_s or 0.0, result.outcome,
            )
        else:
            logger.debug("Step %d: No incident", i)
        time.sleep(0.01)


def main() -> int:
    args = parse_args()
    config = build_config(args)
    stop_event = Event()

    def handle_signal(signum, frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/vector_db", exist_ok=True)
    with ASHAIROrchestrator(config) as orchestrator:
        existing = orchestrator.rag_engine.load()
        if existing:
            logger.info("Loaded RAG DB: %d records", orchestrator.rag_engine.vector_db.record_count())
        if args.dry_run:
            run_dry_run_mode(
                orchestrator=orchestrator,
                domain=args.domain,
                max_steps=args.max_incidents or 100,
                stop_event=stop_event,
            )
        else:
            run_live_mode(
                orchestrator=orchestrator,
                domain=args.domain,
                config=config,
                poll_interval=args.poll_interval,
                stop_event=stop_event,
                max_incidents=args.max_incidents,
            )
    logger.info("ASHAIR runtime shut down cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
