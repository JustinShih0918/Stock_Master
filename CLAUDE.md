# Stock Master

## Project intent

Build a Taiwan-stock and ETF portfolio research system, beginning with paper trading. The initial scale should support a competition-style portfolio starting with NT$1,000,000,000 and daily decision/report artifacts.

## Working rules

- Keep all trading simulated. Do not add or call a real brokerage order endpoint.
- Treat model output as a proposed decision. Validate it against deterministic portfolio, cash, position, and risk rules before recording simulated orders.
- Keep market data, strategy logic, portfolio accounting, execution simulation, and reporting as separate components with explicit interfaces.
- Record inputs, model/provider/model identifier, prompts or strategy version, proposed actions, validation outcomes, fills, and valuation timestamps so a run can be reproduced.
- Use Asia/Taipei for exchange dates and report timestamps. Store timestamps with timezone information.
- Never write API keys or authentication material into source files, logs, or committed fixtures. Read provider credentials from environment variables.
- Prefer deterministic behavior and explicit assumptions. Clearly identify synthetic or stale market data.

## Current development environment

- Ubuntu 24.04 development container with Python, Node.js, Git, PostgreSQL client, and Claude Code.
- PostgreSQL is available at `db:5432` from the dev container and `localhost:5432` from the host.
- NVIDIA hosted-model settings are passed as `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, and `NVIDIA_MODEL`; no strategy harness is implemented yet.
