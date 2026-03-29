# Customer Support Resolution (OpenEnv)

This environment is designed to distinguish between **surface-level fluency** and **true problem-solving ability** in AI customer support agents.

Build for evaluation of autonomous agents in a realistic e-commerce (NovaMart) setting.

## 🚀 Overview

AI agents receive customer support tickets containing query text, sentiment, order history, and company policies. The agent must provide a response that resolves the customer's issues while adhering to strict logical and safety constraints.

---

## 🛡️ Why This Environment is Robust

Unlike naive benchmarks, this system uses a **Deterministic Grading Engine** to prevent reward inflation and ensure reproducible results.

- **Correctness Gating**: Substantially reduces rewards if core resolutions are missed, regardless of tone. 
- **Anti-Bluffing Logic**: Detects and penalizes "helpfulness-sounding" responses that contain no actual resolution steps.
- **Ordering Constraints**: Evaluates logical priority (e.g., Security reset MUST precede financial refund).
- **Audit-Ready Reasoning**: Returns structural feedback for every scoring dimension for transparency.

---

## 📋 Task Scenarios

| Task ID | Difficulty | Scenario |
| :--- | :--- | :--- |
| `easy_001` | Easy | Simple shipping delay apology and status update. |
| `med_001` | Medium | Multi-part request: refund for one item, expedited shipping for another. |
| `hard_001` | Hard | Repeated service failures, damaged items, and chargeback threats. |
| `hard_002` | Hard | **Security Breach**: unauthorized access requiring safe recovery before billing. |

---

## ⚙️ How to Run

### Local
```bash
pip install -r requirements.txt
python inference.py
```
*Requires `HF_TOKEN` in environment for inference.*

### Docker
```bash
docker build -t support-env .
docker run -e HF_TOKEN=your_token_here support-env
```

### Hugging Face
> Deploy this repository as a **Docker Space**. 
> Set `HF_TOKEN`, `API_BASE_URL` (optional), and `MODEL_NAME` (optional) in **Settings > Variables and Secrets**.

---

## 🏗️ Project Structure
- `environment.py`: OpenEnv Gymnasium implementation.
- `grading.py`: Deterministic scoring logic and gating.
- `tasks.py`: Structured task definitions and scenarios.
- `inference.py`: Production-grade LLM evaluation loop.
- `openenv.yaml`: OpenEnv manifest for automated validation.

---
**License**: MIT
