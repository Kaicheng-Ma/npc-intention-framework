# NPC Intention and Behavior Framework

Research prototypes for turning character profiles, world context, player input and conversation history into NPC intentions and responses. This is the NPC component of an AI-driven multi-agent world simulation project, not a complete game engine.

## Contents
- `experiments/single_call`: a single-request story workflow.
- `experiments/multi_call/story`: staged context analysis, candidate intentions, optional tradeoff and response generation.
- `experiments/multi_call/border`: a separate border-inspector tradeoff scenario.
- `data/story` and `data/border`: fictional scenario inputs. Original language and prompt behavior are preserved.
- Personality scoring weights and historical model-distribution training utilities are excluded; this distribution focuses on authored NPC workflows.

## Setup
Use Python 3.11 or later. Install `requirements.txt` in a suitable environment. Copy `config/qwen.env.example` to `config/qwen.env`, then fill in your own API key, official OpenAI-compatible Qwen base URL ending in `/compatible-mode/v1`, and model ID. No credentials are included.

Run an experiment from the repository root, for example:
```sh
python experiments/single_call/Single_Optimized_Version1.py
```
Enter `exit` at the player prompt to stop. Dialogue uses a paid external API. Scenarios are selected by the filenames in each entry point, not by a runtime selector.

## Scope and limitations
The staged workflows use ordinary Python orchestration despite historical filenames containing LangGraph. Output constraints and the action system remain incomplete. Candidate parsing, trigger handling and unbounded retries need further validation. The broader team narrative, memory and player services are not included. OCEAN scoring is a separate exploratory prototype, not a verified constraint on generated dialogue.

This curated copy received Python syntax and JSON checks only; no live API requests, model training or end-to-end quality evaluation were performed during publication.
