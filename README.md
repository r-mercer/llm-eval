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
eval config add-model --name openai --provider openai --api-key $OPENAI_KEY
```

4) Adding tasks
- Define tasks to evaluate: prompts, expected outputs, or evaluation targets.
- Example:
```
eval task add --name sentiment-classification --input-text 'Is this sentiment positive, negative, or neutral? ...'
```

5) Running an evaluation
- Create an experiment and run it against configured models.
```
eval experiment create --name my-first-eval
eval run experiment --id <exp-id>
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
- config list-models
- config list-rubrics
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
eval config init
eval config add-model --name openai --provider openai --api-key $OPENAI_API_KEY
```

- Create a task and import prompts:
```
eval task add --name queuing-task --input-text-file prompts.json
eval task import prompts.json
```

- Create an experiment and run it:
```
eval experiment create --name baseline-eval
eval run experiment --id 1
```

- Show results and export:
```
eval results show --experiment-id 1
eval results export --format json --output results.json
```

## Development

- Setup
- Running tests
- Adding new models
- Contributing guidelines

## Licensing

This project is licensed under the MIT License. See LICENSE for details.
