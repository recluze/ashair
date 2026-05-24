from __future__ import annotations
import asyncio
import logging
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROTOCOL_MODBUS = "modbus"
PROTOCOL_DNP3 = "dnp3"
PROTOCOL_SYSLOG = "syslog"
PROTOCOL_PCAP = "pcap"
PROTOCOL_CSV = "csv"


@dataclass
class RawTelemetryRecord:
    source_protocol: str
    timestamp: datetime
    raw_payload: Any
    source_address: Optional[str] = None
    channel: str = "unknown"


@dataclass
class NormalizedTelemetryRecord:
    timestamp: datetime
    channel: str
    variables: Dict[str, Any]
    source_protocol: str
    source_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModbusTCPPoller:
    def __init__(
        self,
        host: str,
        port: int,
        register_map: Dict[str, Dict[str, Any]],
        poll_interval_s: float = 1.0,
        output_queue: Optional[Queue] = None,
    ):
        self.host = host
        self.port = port
        self.register_map = register_map
        self.poll_interval_s = poll_interval_s
        self.output_queue = output_queue or Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                readings = self._read_registers()
                record = RawTelemetryRecord(
                    source_protocol=PROTOCOL_MODBUS,
                    timestamp=datetime.now(timezone.utc),
                    raw_payload=readings,
                    source_address=f"{self.host}:{self.port}",
                    channel="ot",
                )
                self.output_queue.put(record)
            except Exception as exc:
                logger.error("ModbusTCPPoller: poll error: %s", exc)
            self._stop_event.wait(self.poll_interval_s)

    def _read_registers(self) -> Dict[str, float]:
        try:
            from pymodbus.client import ModbusTcpClient
            client = ModbusTcpClient(self.host, port=self.port)
            client.connect()
            readings = {}
            for sensor_name, reg_info in self.register_map.items():
                address = reg_info["address"]
                unit_id = reg_info.get("unit_id", 1)
                scale = reg_info.get("scale", 0.01)
                result = client.read_holding_registers(address, count=1, slave=unit_id)
                if not result.isError():
                    readings[sensor_name] = result.registers[0] * scale
            client.close()
            return readings
        except ImportError:
            logger.warning("pymodbus not available; returning empty readings")
            return {}


class SyslogReceiver(socketserver.UDPServer):
    def __init__(
        self,
        host: str,
        port: int,
        output_queue: Queue,
    ):
        self.output_queue = output_queue
        super().__init__((host, port), self._make_handler())

    def _make_handler(self):
        queue_ref = self.output_queue

        class SyslogHandler(socketserver.BaseRequestHandler):
            def handle(self):
                data = self.request[0].decode("utf-8", errors="replace")
                record = RawTelemetryRecord(
                    source_protocol=PROTOCOL_SYSLOG,
                    timestamp=datetime.now(timezone.utc),
                    raw_payload=data,
                    source_address=str(self.client_address[0]),
                    channel="it",
                )
                queue_ref.put(record)

        return SyslogHandler

    def start_background(self):
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread


class CSVBatchIngester:
    def __init__(self, output_queue: Optional[Queue] = None):
        self.output_queue = output_queue or Queue()

    def ingest_swat(self, filepath: str, chunk_size: int = 1000) -> List[RawTelemetryRecord]:
        records = []
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            for _, row in chunk.iterrows():
                ts = self._parse_swat_timestamp(row.get("Timestamp", ""))
                payload = row.to_dict()
                record = RawTelemetryRecord(
                    source_protocol=PROTOCOL_CSV,
                    timestamp=ts,
                    raw_payload=payload,
                    channel="swat",
                )
                records.append(record)
        return records

    def ingest_hai(self, filepath: str, chunk_size: int = 1000) -> List[RawTelemetryRecord]:
        records = []
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            for _, row in chunk.iterrows():
                ts = self._parse_hai_timestamp(row.get("timestamp", ""))
                payload = row.to_dict()
                record = RawTelemetryRecord(
                    source_protocol=PROTOCOL_CSV,
                    timestamp=ts,
                    raw_payload=payload,
                    channel="hai",
                )
                records.append(record)
        return records

    def ingest_ics_nad(self, filepath: str, chunk_size: int = 1000) -> List[RawTelemetryRecord]:
        records = []
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            for _, row in chunk.iterrows():
                ts = datetime.now(timezone.utc)
                payload = row.to_dict()
                record = RawTelemetryRecord(
                    source_protocol=PROTOCOL_CSV,
                    timestamp=ts,
                    raw_payload=payload,
                    channel="ics_nad",
                )
                records.append(record)
        return records

    def _parse_swat_timestamp(self, raw: str) -> datetime:
        for fmt in ("%d/%m/%Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                pass
        return datetime.now(timezone.utc)

    def _parse_hai_timestamp(self, raw: str) -> datetime:
        try:
            return datetime.fromisoformat(str(raw)).replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)


class TelemetryNormalizer:
    SWAT_SENSOR_COLUMNS = [
        "FIT101", "LIT101", "MV101", "P101", "P102",
        "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
        "P201", "P203", "P205", "P206",
        "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304",
        "P301", "P302",
        "AIT401", "AIT402",
    ]
    HAI_SENSOR_COLUMNS = [
        "boiler_steam_pressure", "boiler_temperature", "turbine_speed",
        "turbine_output_power", "water_level", "feed_water_flow", "condenser_pressure",
    ]
    ICS_NAD_FEATURE_COLUMNS = [
        "plc_response_time_ms", "network_throughput_mbps", "packet_loss_rate",
        "src_ip", "dst_ip", "protocol", "attack_type",
    ]

    def normalize(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        if record.channel == "swat":
            return self._normalize_swat(record)
        elif record.channel == "hai":
            return self._normalize_hai(record)
        elif record.channel == "ics_nad":
            return self._normalize_ics_nad(record)
        elif record.source_protocol == PROTOCOL_MODBUS:
            return self._normalize_modbus(record)
        elif record.source_protocol == PROTOCOL_SYSLOG:
            return self._normalize_syslog(record)
        else:
            return NormalizedTelemetryRecord(
                timestamp=record.timestamp,
                channel=record.channel,
                variables=record.raw_payload if isinstance(record.raw_payload, dict) else {},
                source_protocol=record.source_protocol,
                source_address=record.source_address,
            )

    def _normalize_swat(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        payload = record.raw_payload
        variables = {}
        for col in self.SWAT_SENSOR_COLUMNS:
            if col in payload:
                try:
                    variables[col] = float(str(payload[col]).strip())
                except (ValueError, TypeError):
                    variables[col] = payload[col]
        label_raw = str(payload.get("Normal/Attack", "Normal")).strip()
        label = "Attack" if "attack" in label_raw.lower() else "Normal"
        return NormalizedTelemetryRecord(
            timestamp=record.timestamp,
            channel="swat",
            variables=variables,
            source_protocol=record.source_protocol,
            metadata={"label": label, "raw_label": label_raw},
        )

    def _normalize_hai(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        payload = record.raw_payload
        variables = {}
        for col in self.HAI_SENSOR_COLUMNS:
            if col in payload:
                try:
                    variables[col] = float(payload[col])
                except (ValueError, TypeError):
                    variables[col] = payload[col]
        attack_flag = payload.get("attack", 0)
        label = "Attack" if int(attack_flag) != 0 else "Normal"
        return NormalizedTelemetryRecord(
            timestamp=record.timestamp,
            channel="hai",
            variables=variables,
            source_protocol=record.source_protocol,
            metadata={"label": label, "attack_type": payload.get("attack_type", "")},
        )

    def _normalize_ics_nad(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        payload = record.raw_payload
        variables = {}
        for col in self.ICS_NAD_FEATURE_COLUMNS:
            if col in payload:
                variables[col] = payload[col]
        label_raw = str(payload.get("label", "Benign")).strip()
        label = "Attack" if label_raw.lower() not in ("benign", "normal", "0") else "Normal"
        return NormalizedTelemetryRecord(
            timestamp=record.timestamp,
            channel="ics_nad",
            variables=variables,
            source_protocol=record.source_protocol,
            metadata={"label": label, "attack_category": payload.get("attack_category", "")},
        )

    def _normalize_modbus(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        return NormalizedTelemetryRecord(
            timestamp=record.timestamp,
            channel="ot",
            variables=record.raw_payload if isinstance(record.raw_payload, dict) else {},
            source_protocol=PROTOCOL_MODBUS,
            source_address=record.source_address,
        )

    def _normalize_syslog(self, record: RawTelemetryRecord) -> NormalizedTelemetryRecord:
        return NormalizedTelemetryRecord(
            timestamp=record.timestamp,
            channel="it",
            variables={"raw_log": record.raw_payload},
            source_protocol=PROTOCOL_SYSLOG,
            source_address=record.source_address,
        )
