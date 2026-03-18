# LLM Evaluation Framework

Tagline
- A CLI tool for evaluating and comparing LLM outputs with configurable rubrics, pairwise comparisons, and ranking systems.

## Features
- Model-agnostic evaluation (OpenAI, Anthropic, Ollama, LM Studio, etc.)
- Pairwise comparisons with Chain-of-Thought reasoning
- Leaderboard-style rankings with Elo rating system
- Configurable weighted rubrics
- PostgreSQL-backed experiment tracking
- JSON/CSV export

## Quick Start

1) Installation
- Ensure you have Python 3.8+ installed.
- Install in editable mode (assumes you have a working Python package layout with setup.py or pyproject.toml):
```
pip install -e .
```

2) Database setup
- Install and configure PostgreSQL.
- Create a database and user, then set up the schema via the CLI or migrations (see Configuration and Architecture sections).

3) Adding models
- Add a model definition for each LLM provider you want to evaluate (OpenAI, Anthropic, Ollama, etc.).
- Example (pseudo):
```
llmeval config add-model --name openai --type openai --api-key $OPENAI_KEY
```

4) Adding tasks
- Define tasks to evaluate: prompts, expected outputs, or evaluation targets.
- Example:
```
llmeval task add --name sentiment-classification --prompt 'Is this sentiment positive, negative, or neutral? ...'
```

5) Running an evaluation
- Create an experiment and run it against configured models.
```
llmeval experiment create --name my-first-eval
llmeval run experiment --id <exp-id>
```

## Configuration

- Environment variables (DB_*)
- Example:
```
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=llmeval
export DB_USER=llmeval
export DB_PASSWORD=secret
```

- Model configuration
- Rubric format and mappings

## CLI Commands Reference
- config init
- config add-model
- config add-rubric
- list-models
- list-rubrics
- task add
- task list
- task import
- experiment create
- experiment list
- experiment status
- run experiment
- run compare
- results show
- results export

Notes:
- The exact command names and flags may vary with release; consult the --help output for the current CLI.

## Architecture

- Database schema overview
- Evaluation flow
- Data model relationships: experiments, tasks, rubrics, models, results, leaderboard

## Examples

- Initialize config and add a model:
```
llmeval config init
llmeval config add-model --name openai --type openai --api-key $OPENAI_API_KEY
```

- Create a task and import prompts:
```
llmeval task add --name queuing-task --prompt-file prompts.json
llmeval task import prompts.json
```

- Create an experiment and run it:
```
llmeval experiment create --name baseline-eval
llmeval run experiment --id 1
```

- Show results and export:
```
llmeval results show --experiment-id 1
llmeval results export --format json --output results.json
```

## Development

- Setup
- Running tests
- Adding new models
- Contributing guidelines

## Licensing

This project is licensed under the MIT License. See LICENSE for details.
