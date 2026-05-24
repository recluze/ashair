# ASHAIR: Agentic Self-Healing and Automated Incident Response

Implementation of the ASHAIR framework from:
> *"ASHAIR: Agentic Self-Healing and Automated Incident Response for Cyber-Physical Systems Using LLM-Driven Multi-Agent Frameworks"*
> Sohail Khan, Mohammad Nauman — Effat University, Jeddah

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --port 8000
```

Set environment variables:

```bash
export VLLM_ENDPOINT=http://localhost:8000/v1
export ZT_SECRET=your-secret-key
export MODBUS_HOST=192.168.1.100
```

---

## Datasets

| Dataset   | Coverage |
|-----------|----------|
| SWaT      | Physical-layer FDI (water treatment) |
| HAI 21.03 | Process diversity (turbine/boiler FDI) |
| ICS-NAD   | Network-layer FDI + DDoS |

---

## Usage

### 1. Pre-populate the RAG vector database

```bash
python training/rag_indexer.py \
    --swat  data/swat_dataset.csv \
    --hai   data/hai_21_03.csv \
    --ics-nad data/ics_nad_features.csv \
    --split train
```

### 2. Run the full evaluation

```bash
python run_evaluation.py \
    --swat  data/swat_dataset.csv \
    --hai   data/hai_21_03.csv \
    --ics-nad data/ics_nad_features.csv
```

### 3. Live / dry-run deployment

```bash
# Dry run with synthetic telemetry
python pipeline_runner.py --domain swat --dry-run --max-incidents 100

