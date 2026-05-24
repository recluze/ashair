from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.ingestion import NormalizedTelemetryRecord


class TelemetryNarrativeGenerator:
    def generate(self, record: NormalizedTelemetryRecord) -> str:
        if record.channel == "swat":
            return self._swat_narrative(record)
        elif record.channel == "hai":
            return self._hai_narrative(record)
        elif record.channel == "ics_nad":
            return self._ics_nad_narrative(record)
        elif record.channel == "ot":
            return self._ot_narrative(record)
        elif record.channel == "it":
            return self._it_narrative(record)
        else:
            return self._generic_narrative(record)

    def _swat_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        vars_ = record.variables
        lit101 = vars_.get("LIT101", "N/A")
        fit101 = vars_.get("FIT101", "N/A")
        mv101 = vars_.get("MV101", "N/A")
        p101 = vars_.get("P101", "N/A")
        ait201 = vars_.get("AIT201", "N/A")
        fit201 = vars_.get("FIT201", "N/A")
        lit301 = vars_.get("LIT301", "N/A")
        fit301 = vars_.get("FIT301", "N/A")
        anomalies = self._detect_swat_anomalies(vars_)
        anomaly_str = f" ANOMALIES DETECTED: {'; '.join(anomalies)}." if anomalies else " All readings within expected bounds."
        return (
            f"[{ts}] SWaT Water Treatment Telemetry: "
            f"Stage 1 — Tank LIT101={lit101}m (nominal 0.500–0.800m), "
            f"Inlet flow FIT101={fit101}m³/h, Valve MV101={'OPEN' if mv101 == 1 else 'CLOSED'}, "
            f"Pump P101={'ON' if p101 == 1 else 'OFF'}. "
            f"Stage 2 — Chemical dosing AIT201={ait201}, Flow FIT201={fit201}m³/h. "
            f"Stage 3 — Tank LIT301={lit301}m, Filter flow FIT301={fit301}m³/h."
            f"{anomaly_str}"
        )

    def _hai_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        vars_ = record.variables
        boiler_temp = vars_.get("boiler_temperature", "N/A")
        boiler_pressure = vars_.get("boiler_steam_pressure", "N/A")
        turbine_speed = vars_.get("turbine_speed", "N/A")
        turbine_power = vars_.get("turbine_output_power", "N/A")
        water_level = vars_.get("water_level", "N/A")
        anomalies = self._detect_hai_anomalies(vars_)
        anomaly_str = f" ANOMALIES DETECTED: {'; '.join(anomalies)}." if anomalies else " All process variables nominal."
        return (
            f"[{ts}] HAI Hardware-in-Loop Telemetry: "
            f"Boiler — temperature={boiler_temp}°C (nominal 200–900°C), "
            f"steam pressure={boiler_pressure}bar. "
            f"Turbine — speed={turbine_speed}RPM (nominal 0–3600), "
            f"output power={turbine_power}kW. "
            f"Water system — level={water_level} (nominal 0.20–0.90)."
            f"{anomaly_str}"
        )

    def _ics_nad_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        vars_ = record.variables
        src_ip = vars_.get("src_ip", "unknown")
        dst_ip = vars_.get("dst_ip", "unknown")
        protocol = vars_.get("protocol", "unknown")
        throughput = vars_.get("network_throughput_mbps", "N/A")
        response_time = vars_.get("plc_response_time_ms", "N/A")
        packet_loss = vars_.get("packet_loss_rate", "N/A")
        attack_cat = record.metadata.get("attack_category", "")
        attack_str = f" Attack category: {attack_cat}." if attack_cat else ""
        return (
            f"[{ts}] ICS Network Telemetry: "
            f"Flow {src_ip} → {dst_ip} via {protocol}. "
            f"Throughput={throughput}Mbps, "
            f"PLC response time={response_time}ms (threshold 500ms), "
            f"Packet loss={packet_loss}."
            f"{attack_str}"
        )

    def _ot_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        var_str = ", ".join(f"{k}={v}" for k, v in record.variables.items())
        return f"[{ts}] OT Process Telemetry (Modbus): {var_str}"

    def _it_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        raw_log = record.variables.get("raw_log", "")
        return f"[{ts}] IT System Log from {record.source_address}: {raw_log}"

    def _generic_narrative(self, record: NormalizedTelemetryRecord) -> str:
        ts = record.timestamp.isoformat()
        return f"[{ts}] Telemetry ({record.channel}): {json.dumps(record.variables)}"

    def _detect_swat_anomalies(self, vars_: Dict[str, Any]) -> List[str]:
        anomalies = []
        swat_bounds = {
            "LIT101": (0.5, 0.8), "FIT101": (0.0, 2.5),
            "LIT301": (0.5, 0.8), "FIT301": (0.0, 2.5),
            "AIT401": (0.0, 0.05), "AIT402": (0.0, 0.05),
        }
        for sensor, (lo, hi) in swat_bounds.items():
            val = vars_.get(sensor)
            if val is None:
                continue
            try:
                fval = float(val)
                if fval < lo or fval > hi:
                    anomalies.append(f"{sensor}={fval:.4f} outside [{lo}, {hi}]")
            except (ValueError, TypeError):
                pass
        return anomalies

    def _detect_hai_anomalies(self, vars_: Dict[str, Any]) -> List[str]:
        anomalies = []
        hai_bounds = {
            "boiler_temperature": (200.0, 900.0),
            "boiler_steam_pressure": (0.0, 10.0),
            "turbine_speed": (0.0, 3600.0),
            "water_level": (0.2, 0.9),
        }
        for sensor, (lo, hi) in hai_bounds.items():
            val = vars_.get(sensor)
            if val is None:
                continue
            try:
                fval = float(val)
                if fval < lo or fval > hi:
                    anomalies.append(f"{sensor}={fval:.2f} outside [{lo}, {hi}]")
            except (ValueError, TypeError):
                pass
        return anomalies
