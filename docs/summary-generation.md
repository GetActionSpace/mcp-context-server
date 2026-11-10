# Summary Generation Guide

## Introduction

Summary generation automatically creates concise, dense summaries for each stored context entry using a local or cloud LLM. Summaries are stored alongside the full text and returned in all search tool results (`search_context`, `semantic_search_context`, `fts_search_context`, `hybrid_search_context`), giving LLM agents more actionable information per token when browsing large context collections.

**Key benefit:** All search tools return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 300 characters). With summary generation enabled, the `summary` field is populated with a concise LLM-generated summary of the full entry (token limit controlled by `SUMMARY_MAX_TOKENS`), capturing key topics, decisions, and action items that help an agent determine relevance without fetching the full entry.

This feature is **enabled by default** when the `summary-ollama` extra is installed (included in the recommended setup).

## Summary Providers

The server supports three summary providers via LangChain integration:

| Provider             | Default Model    | Cost            | Best For                     |
|----------------------|------------------|-----------------|------------------------------|
| **Ollama** (default) | qwen3:0.6b       | Free (local)    | Development, privacy-focused |
| **OpenAI**           | gpt-5.4-nano     | Pay-per-use API | Production, high quality     |
| **Anthropic**        | claude-haiku-4-5 | Pay-per-use API | Production, high quality     |

Select a provider via the `SUMMARY_PROVIDER` environment variable.

## Prerequisites

- **Python**: 3.12+ (already required by MCP Context Server)
- **Ollama** (for default provider): Installed from [ollama.com/download](https://ollama.com/download)
- **RAM**: 2GB minimum for `qwen3:0.6b`; 4GB recommended for `qwen3:1.7b` and 16GB for `qwen3:8b`

## Installation

### Step 1: Install Provider Dependencies

Each provider has its own optional dependency group:

```bash
# Ollama provider (default)
uv sync --extra summary-ollama

# OpenAI provider
uv sync --extra summary-openai

# Anthropic provider
uv sync --extra summary-anthropic
```

### Step 2: Provider-Specific Setup

See the provider-specific sections below for model and credential requirements.

## Provider Configuration

### Ollama (Default)

Ollama runs summary models locally with no API costs. The default model `qwen3:0.6b` is optimized for fast summaries with minimal resource requirements.

#### Setup

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download)

2. **Pull the summary model**:
   ```bash
   ollama pull qwen3:0.6b
   ```

3. **Verify**:
   ```bash
   ollama list
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-ollama,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "ollama",
        "SUMMARY_MODEL": "qwen3:0.6b"
      }
    }
  }
}
```

#### Environment Variables

| Variable                            | Default      | Description                                                                                                                                            |
|-------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION`         | `true`       | Enable/disable summary generation                                                                                                                      |
| `SUMMARY_PROVIDER`                  | `ollama`     | Set to `ollama`                                                                                                                                        |
| `SUMMARY_MODEL`                     | `qwen3:0.6b` | Ollama model name (see model table below)                                                                                                              |
| `SUMMARY_MAX_TOKENS`                | `4000`       | Token budget per call (50-16384), sent as `num_predict`. With [reasoning off](#reasoning-off), it covers only the summary text                         |
| `SUMMARY_TIMEOUT_S`                 | `240.0`      | Timeout in seconds for summary generation API calls                                                                                                    |
| `SUMMARY_RETRY_MAX_ATTEMPTS`        | `5`          | Maximum retry attempts on transient errors                                                                                                             |
| `SUMMARY_RETRY_BASE_DELAY_S`        | `3.0`        | Base delay in seconds between retries (exponential backoff)                                                                                            |
| `SUMMARY_MAX_CONCURRENT`            | `2`          | Maximum concurrent calls against the summary model (1-20). Shared budget covering both the flat document summary and the index_tree per-node summaries |
| `SUMMARY_MIN_CONTENT_LENGTH`        | `500`        | Minimum text length (characters) to trigger summary generation. 0 = always generate                                                                    |
| `SUMMARY_PROMPT`                    | (built-in)   | Custom system prompt. Overrides both source-specific defaults. See [Custom Prompt](#custom-prompt)                                                     |
| `SUMMARY_OLLAMA_NUM_CTX`            | `32768`      | Ollama context window in tokens (512-2097152). Must accommodate input text + prompt + output budget                                                    |
| `SUMMARY_OLLAMA_TRUNCATE`           | `false`      | Truncation mode: false (default) returns error when context exceeded, true enables silent truncation                                                   |
| `SUMMARY_OLLAMA_PRESENCE_PENALTY`   | _(unset)_    | Presence penalty (0-2) sent with every call. Unset: `1.5` for a `qwen3` model, none for any other; see [Reasoning Off](#reasoning-off)                 |

#### Reasoning Off

The Ollama provider sends every summary call, the document summary and each per-section summary alike, with the model's reasoning turned off: LangChain's `reasoning=False`, which sends Ollama's `think: false` request parameter. A model that can reason, such as Qwen3, then writes the summary straight away, so `SUMMARY_MAX_TOKENS` bounds the summary text instead of a hidden reasoning trace followed by the summary.

Ollama turns reasoning off only for models that allow it. The hybrid Qwen3 builds `qwen3:0.6b`, `qwen3:1.7b` and `qwen3:8b` allow it. The builds the Ollama library currently serves as `qwen3:4b` and `qwen3:30b` reason on every call, so they are unsuitable while reasoning is off: Ollama returns their reasoning and the closing `</think>` tag as part of the answer, the stored summary starts with the model's reasoning, and the reasoning counts against `SUMMARY_MAX_TOKENS`. With any model, a call that uses the whole budget and returns no text fails: a failed document summary means the server does not store the entry (see [How It Works](#how-it-works)), or fails the attempt of a deferred write's generation job (see [Deferred Summaries](#deferred-summaries)), and a failed per-section summary leaves that section without a summary while the entry is stored.

Reasoning off also removes the planning step that kept a small model from copying its input. Given low-entropy text, such as a document made of one repeated token, `qwen3:0.6b` echoes the text until the `SUMMARY_MAX_TOKENS` budget runs out, every attempt then exceeds `SUMMARY_TIMEOUT_S`, and the store fails once the retries are spent; a sampling temperature does not stop it. The Qwen3 model card's remedy is a presence penalty, so the provider sends `presence_penalty: 1.5` with every call to a `qwen3` model and no penalty to a model of another family, whose authors give no such guidance. `SUMMARY_OLLAMA_PRESENCE_PENALTY` (0-2) replaces that default for whatever model is configured, and `0` disables the penalty. The penalty lowers the odds of a token the output already contains, so a summary repeats a term a little less often; the model card notes that higher values occasionally mix languages. Each per-section summary runs under its own output budget, `INDEX_TREE_NODE_SUMMARY_MAX_TOKENS` (default 120, sized for a headline), and an answer that runs to that budget drops the section's summary instead of being stored cut off.

#### Qwen3 Model Options (Ollama)

The Qwen3 family offers a range of sizes for different resource constraints and quality requirements:

| Model        | RAM Required | Quality   | Speed     | Notes                                          |
|--------------|--------------|-----------|-----------|------------------------------------------------|
| `qwen3:0.6b` | ~2GB         | Basic     | Fastest   | **Default**. Lightweight, minimal resources    |
| `qwen3:1.7b` | ~4GB         | Good      | Fast      | Higher quality, good balance for most uses     |
| `qwen3:8b`   | ~16GB        | Best      | Slower    | Highest quality, requires dedicated hardware   |

**Recommendation:** Start with `qwen3:0.6b` (default). Upgrade to `qwen3:1.7b` if summary quality is insufficient for your use case, or to `qwen3:8b` for the highest quality on dedicated hardware.

Pull any alternative model before use:
```bash
ollama pull qwen3:1.7b
```

### OpenAI

OpenAI provides high-quality summaries via API with no local hardware requirements.

#### Setup

1. **Get API key** from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

2. **Install dependencies**:
   ```bash
   uv sync --extra summary-openai
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-openai,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "openai",
        "SUMMARY_MODEL": "gpt-5.4-nano",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

#### Environment Variables

| Variable                          | Default        | Description                                                                                                                                          |
|-----------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SUMMARY_PROVIDER`                | -              | Set to `openai`                                                                                                                                      |
| `SUMMARY_MODEL`                   | `gpt-5.4-nano` | OpenAI model name                                                                                                                                    |
| `OPENAI_API_KEY`                  | -              | **Required**: OpenAI API key                                                                                                                         |
| `SUMMARY_OPENAI_REASONING_EFFORT` | `low`          | Reasoning effort for reasoning models. Valid: gpt-5: low/medium/high; gpt-5.1+: none/low/medium/high/xhigh. Default `low` is universally valid       |

### Anthropic

Anthropic's Claude models offer high-quality summaries with strong instruction-following.

#### Setup

1. **Get API key** from [console.anthropic.com](https://console.anthropic.com)

2. **Install dependencies**:
   ```bash
   uv sync --extra summary-anthropic
   ```

#### Configuration

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-anthropic,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "SUMMARY_PROVIDER": "anthropic",
        "SUMMARY_MODEL": "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

#### Environment Variables

| Variable                   | Default                     | Description                                                                                                                       |
|----------------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `SUMMARY_PROVIDER`         | -                           | Set to `anthropic`                                                                                                                |
| `SUMMARY_MODEL`            | `claude-haiku-4-5-20251001` | Anthropic model name                                                                                                              |
| `ANTHROPIC_API_KEY`        | -                           | **Required**: Anthropic API key                                                                                                   |
| `SUMMARY_ANTHROPIC_EFFORT` | _(none)_                    | Effort level for Claude models. Valid: `max`, `high`, `medium`, `low`. Not sent by default (required for Haiku 4.5 compatibility) |

## Source-Aware Summarization

Summary prompts are dynamically selected based on the `source` field of the context entry being summarized. The model sees instructions tailored to the specific source type -- it never sees instructions for the other source type.

- **User messages** (`source='user'`): The prompt focuses on capturing user intent, requirements, constraints, and directives.
- **Agent reports** (`source='agent'`): The prompt focuses on key findings, decisions, deliverables, and technical specifics while omitting process metadata.

Both prompts share common requirements (single paragraph, English output, no labels/prefixes) but differ in their focus instructions. Summaries are always generated in English regardless of input language.

## Custom Prompt

The server ships with carefully engineered source-specific summarization prompts. For most use cases, the default prompts work well. You can override them with a single custom prompt via the `SUMMARY_PROMPT` environment variable. When set, the custom prompt is used for both source types (source-specific logic is bypassed).

### Default Prompts

The built-in prompts (from `app/summary/instructions.py`) use source-specific instructions with shared base requirements:

**User messages** -- focuses on intent, requirements, and directives:
```text
/no_think
You are an expert summarizer for a context storage system used by AI agents.
What follows was written by a human user: a message, a note, or a document. Your task is to produce
a single, dense paragraph that captures its essential meaning.
...
```

**Agent reports** -- focuses on findings, decisions, and deliverables:
```text
/no_think
You are an expert summarizer for a context storage system used by AI agents.
The following text is a work report generated by an AI agent. Your task is to produce
a single, dense paragraph that captures the essential meaning of the agent's report.
...
```

**Design notes:**
- `/no_think` is Qwen3's in-prompt switch for skipping reasoning; the Ollama provider also turns reasoning off in the request itself (see [Reasoning Off](#reasoning-off))
- Zero-shot format maximizes input token budget on resource-constrained models
- Single-paragraph constraint is easiest for small models to follow consistently
- Negative constraints (`Do not...`) prevent common small-model failure modes
- A model that answers without reasoning opens with the most salient noun phrase of the task statement, so no prompt names the user, the text, or the section as the thing to describe; the per-section prompt asks for a headline-style noun phrase, because such a model copies a sentence from the section when asked for a sentence and copies a worked example when given one
- Summaries are always in English regardless of input language

### Overriding the Prompt

Set `SUMMARY_PROMPT` to your custom system message:

```json
{
  "env": {
    "SUMMARY_PROMPT": "You are a technical documentation summarizer. Produce a single sentence capturing the main purpose, key entities, and outcome. Output only the sentence."
  }
}
```

**Important notes:**
- The prompt is used as the **system message**. The text to summarize is passed separately as the user message (AS-IS, without any prefix).
- When `SUMMARY_PROMPT` is set, it overrides BOTH source-specific prompts (user and agent get the same custom prompt).
- An empty string or whitespace-only value falls back to the source-specific default prompts (unlike `MCP_SERVER_INSTRUCTIONS` where empty string disables the feature).
- The Ollama provider turns reasoning off in the request, so a custom prompt needs no `/no_think` there. For Qwen3 served through another provider, such as an OpenAI-compatible endpoint, start your prompt with `/no_think` to skip reasoning and save tokens.

## Cross-Provider Reasoning Limitations

When using reasoning models for summary generation, each provider has specific limitations and controls:

| Provider      | Control Mechanism                 | Limitation                                                                           |
|---------------|-----------------------------------|--------------------------------------------------------------------------------------|
| **Ollama**    | `think: false` on every request   | Only models that allow it; see [Reasoning Off](#reasoning-off)                       |
| **OpenAI**    | `SUMMARY_OPENAI_REASONING_EFFORT` | Controls reasoning token budget. Default `low` (valid across all model generations)  |
| **Anthropic** | `SUMMARY_ANTHROPIC_EFFORT`        | Controls adaptive thinking. Not supported on all models (e.g., Haiku 4.5)            |

### Known Constraints

- **A repetitive document needs the presence penalty (Ollama):** With reasoning off, a `qwen3` model copies low-entropy text until its token budget runs out, so the provider sends the model card's `presence_penalty: 1.5` to that family; another family gets none unless `SUMMARY_OLLAMA_PRESENCE_PENALTY` sets one. See [Reasoning Off](#reasoning-off).
- **Reasoning off needs a model that allows it (Ollama):** A model that reasons on every call keeps reasoning under `think: false`, and Ollama then returns the reasoning as part of the summary ([langchain-ai/langchain#33993](https://github.com/langchain-ai/langchain/issues/33993) reports that case). Use a hybrid model, such as `qwen3:1.7b` or `qwen3:8b`, with this provider; see [Reasoning Off](#reasoning-off).
- **`effort` is not supported on Haiku 4.5 (Anthropic):** The `SUMMARY_ANTHROPIC_EFFORT` setting defaults to `None` (parameter omitted) because Claude Haiku 4.5 does not support the `effort` parameter. Set this only when using models that support adaptive thinking (e.g., Claude Sonnet 4, Claude Opus 4).
- **Temperature goes only to models that accept it (Anthropic):** The Ollama and OpenAI providers request `temperature=0` for consistent output. Claude Opus 4.7 and later, Claude Sonnet 5 and Claude Fable 5 reject sampling parameters with HTTP 400, so the Anthropic provider sends `temperature=0` only when the model profile bundled with `langchain-anthropic` declares temperature support, as it does for Claude Haiku 4.5 and the Claude Sonnet and Opus 4.5 and 4.6 models. Any other model, including one missing from that profile data, receives no temperature and uses its default. Extended thinking with a token budget requires `temperature=1`, so it cannot run on a model that receives `temperature=0`; the `effort` parameter controls adaptive thinking, a different mechanism, and works either way.
- **Non-Qwen3 reasoning models require model-specific approaches:** There is no universal mechanism to disable or control reasoning across all model families. When using a new reasoning model, consult its documentation for the appropriate control mechanism.

## How It Works

1. **On `store_context`**: Summary generation runs in parallel with embedding generation before the database transaction. Both must complete before data is saved. If summary generation fails, the entire operation fails (transactional integrity).
2. **On `store_context_batch`**: Summary generation runs for each entry, beside its embedding generation, before the batch opens any transaction. A failure stores nothing of that entry, and nothing at all in atomic mode.
3. **On `update_context`**: When `text` changes, summary and embedding are regenerated in parallel before the update transaction.
4. **On `update_context_batch`**: Summary regenerated for entries with text changes, before the batch opens any transaction.
5. **Deduplication**: When a deduplicating store (`deduplicate` true, the default) finds that its likely duplicate already has a summary, summary generation is skipped and that summary is reused. A store with `deduplicate=false` always inserts a new entry and generates its summary like any other new entry.
6. **Deferred generation**: A store or text update made with `defer_generation=true` generates no summary in the request; a background job generates it afterward (see [Deferred Summaries](#deferred-summaries)).

### Deferred Summaries

A write with `defer_generation=true` (see [Generation Status](api-reference.md#generation-status)) stores its entry without a summary and leaves the summary, the embeddings and the per-section summaries to a generation job that the server's worker runs in the background:

- **Until the job completes** the entry has no summary: search results carry an empty `summary`, `get_context_by_ids` returns `''` when `GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY=true`, and `navigate_context` reports a null root summary and no node summaries, even with `include_node_summaries=True`. A deferred text update clears the old text's summary and section summaries when it stores the new text, so no read ever pairs the new text with the old summary.
- **The job generates what is missing.** It runs the summary leg when a summary provider is active and the entry has no summary, under the same `SUMMARY_MIN_CONTENT_LENGTH` rule as a synchronous write, so a short text stays without a summary. The section summaries follow the flat summary as they do in a request and share the same `SUMMARY_MAX_CONCURRENT` budget, which the worker shares with every synchronous request of the server.
- **A summary failure fails the job's attempt.** When the summary still fails after its `SUMMARY_RETRY_MAX_ATTEMPTS` provider attempts, the worker discards the attempt's embeddings as well and retries the job after a backoff; after `GENERATION_JOB_MAX_ATTEMPTS` attempts (3 by default) the entry turns `failed` and keeps no summary until a server restart or `mcp-context-server-migrate --embed-missing` runs the job again (see [Retries and Failed Jobs](semantic-search.md#retries-and-failed-jobs)). A section-summary failure never fails the job: the section stays without a summary.

### Minimum Content Length

By default, summary generation is skipped for short text (fewer than 500 characters). Text under 500 characters is adequately served by the 300-character truncated preview returned by all search tools, so a separate LLM-generated summary adds minimal value -- particularly for small models like qwen3:0.6b that tend to produce paraphrases rather than distillations for short inputs.

| Variable                       | Default | Range      | Description                                                      |
|--------------------------------|---------|------------|------------------------------------------------------------------|
| `SUMMARY_MIN_CONTENT_LENGTH`   | `500`   | 0 - 10000  | Minimum text length (characters) to trigger summary generation   |

**Behavior by operation:**

- **`store_context`** / **`store_context_batch`**: Text shorter than the threshold is stored without a summary (`summary` field is empty).
- **`update_context`** / **`update_context_batch`**: When updated text is shorter than the threshold, any existing summary is cleared (set to NULL) since the summary no longer accurately represents the content.
- **Deduplication**: When a deduplicating store (`deduplicate` true) updates a short-text duplicate without generating a summary, any pre-existing summary on the original entry is preserved via SQL `COALESCE`. With `deduplicate=false` the short text is stored as a new entry without a summary.
- **Deferred generation**: The generation job applies the same threshold to the stored text, so a short text written with `defer_generation=true` turns `ready` without a summary.

**Special values:**

- `0` disables the threshold entirely: summaries are always generated regardless of text length.
- The comparison uses strict `<` (text at exactly the threshold length IS summarized).

### Context Length and Truncation Control (Ollama)

When using Ollama for summary generation, context length and truncation behavior are configurable. OpenAI and Anthropic providers handle context limits server-side and return explicit HTTP errors when limits are exceeded.

#### Truncation Behavior by Provider

| Provider        | Truncation Control | Default Behavior                        | Configuration                     |
|-----------------|--------------------|-----------------------------------------|-----------------------------------|
| **Ollama**      | Configurable       | Error on context exceed                 | `SUMMARY_OLLAMA_TRUNCATE=false`   |
| **OpenAI**      | Always error       | Returns HTTP 400 if input exceeds limit | N/A                               |
| **Anthropic**   | Always error       | Returns HTTP 400 if input exceeds limit | N/A                               |

#### Recommended Configuration

For production use, keep truncation **disabled** (default):

```bash
SUMMARY_OLLAMA_TRUNCATE=false       # Default - errors prevent silent quality degradation
```

When truncation is disabled, text length is estimated before calling the Ollama API. If the estimated token count exceeds the available input budget (context window minus output budget minus prompt overhead), an error is raised with actionable guidance.

**Available input budget** = `SUMMARY_OLLAMA_NUM_CTX` - `SUMMARY_MAX_TOKENS` - prompt overhead (~120 tokens for default prompt)

Example error:
```text
ValueError: Text length (15000 chars, ~5000 estimated tokens) may exceed available input budget
(28538 tokens from model spec (32768) capped by SUMMARY_OLLAMA_NUM_CTX (32768),
after reserving 4000 output + ~230 prompt tokens) for model qwen3:0.6b.
Options: 1) Increase SUMMARY_OLLAMA_NUM_CTX,
         2) Set SUMMARY_OLLAMA_TRUNCATE=true to allow silent truncation,
         3) Use a larger-context model.
```

**Note**: The effective context limit for Ollama models is `min(model_max_input_tokens, SUMMARY_OLLAMA_NUM_CTX)`. The `SummaryModelSpec` registry in `app/summary/context_limits.py` contains known model specifications.

#### Context Limits by Model

| Model              | Provider    | Max Input Tokens | Notes                                        |
|--------------------|-------------|------------------|----------------------------------------------|
| qwen3:0.6b         | Ollama      | 32,768           | Default model                                |
| qwen3:1.7b         | Ollama      | 32,768           | Higher quality                               |
| qwen3:4b           | Ollama      | 131,072          | YaRN enabled by default on Ollama            |
| qwen3:8b           | Ollama      | 32,768           | Highest quality for Ollama                   |
| qwen3:14b          | Ollama      | 32,768           | Large model, dedicated hardware required     |
| qwen3:32b          | Ollama      | 32,768           | Largest Ollama model                         |
| gpt-5.4-nano       | OpenAI      | 400,000          | Always returns error on exceed               |
| gpt-5.4-mini       | OpenAI      | 400,000          | Always returns error on exceed               |
| gpt-5.4            | OpenAI      | 400,000          | Always returns error on exceed               |
| claude-haiku-4-5   | Anthropic   | 200,000          | Standard tier; 1M available with beta header |
| claude-sonnet-4    | Anthropic   | 200,000          | Standard tier; 1M available with beta header |
| gpt-5-nano         | OpenAI      | 400,000          | Always returns error on exceed               |
| claude-opus-4-6    | Anthropic   | 200,000          | Always returns error on exceed               |
| claude-sonnet-4-6  | Anthropic   | 200,000          | Always returns error on exceed               |

### Summary in Search Results

The `summary` field appears in all search tool results when available:

```json
{
  "results": [
    {
      "id": "0190abcdef1234567890abcdef123456",
      "thread_id": "project-abc",
      "source": "agent",
      "text_content": "Agent implemented OAuth2 authentication with JWT tokens for the user management API, resolving rate-limiting issues on the /auth/...",
      "summary": "Agent implemented OAuth2 authentication with JWT tokens for the user management API, resolving rate-limiting issues on the /auth/login endpoint.",
      "is_text_content_truncated": true,
      "visibility": "private",
      "is_owner": true,
      "generation_status": "ready",
      "metadata": {"agent_name": "developer", "status": "done"},
      "tags": ["implementation", "auth"]
    }
  ]
}
```

All search tools always return truncated `text_content` (configurable via `SEARCH_TRUNCATION_LENGTH`, default 300 characters) with `is_text_content_truncated` flag. The `summary` field provides a dense LLM-generated summary (controlled by `SUMMARY_MAX_TOKENS`, default 4000 tokens) when summary generation is enabled, or an empty string when disabled or not yet generated, as while an entry's `generation_status` is `pending`. Use `get_context_by_ids` to retrieve the full, untruncated text content.

## Disabling Summary Generation

To disable summary generation entirely:

```json
{
  "env": {
    "ENABLE_SUMMARY_GENERATION": "false"
  }
}
```

When disabled:
- No LLM calls are made at store/update time
- The `summary` field in search results is always an empty string
- No summary provider dependencies are required at startup

**Note:** Like `ENABLE_EMBEDDING_GENERATION`, when `ENABLE_SUMMARY_GENERATION=true` (default) and the required provider package is not installed, the server will NOT start. Set to `false` to run without summary generation.

## Installation with Multiple Features

When combining summary generation with semantic search:

```bash
# Ollama for both embeddings and summaries (recommended)
uv sync --extra embeddings-ollama --extra summary-ollama --extra reranking

# Ollama embeddings + OpenAI summaries
uv sync --extra embeddings-ollama --extra summary-openai --extra reranking
```

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[embeddings-ollama,summary-ollama,reranking]", "mcp-context-server"],
      "env": {
        "ENABLE_SUMMARY_GENERATION": "true",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "qwen3-embedding:0.6b",
        "SUMMARY_PROVIDER": "ollama",
        "SUMMARY_MODEL": "qwen3:0.6b",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

## Troubleshooting

### Summary Not Generated

**Symptom:** `summary` field is an empty string in search results.

**Causes and solutions:**

| Cause                             | Solution                                                                                                                                                                               |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION=false` | Set to `true` and install provider dependencies                                                                                                                                        |
| Provider package not installed    | Run `uv sync --extra summary-ollama` or `summary-openai`                                                                                                                               |
| Ollama not running                | Start Ollama: `ollama serve`                                                                                                                                                           |
| Model not pulled                  | Run `ollama pull qwen3:0.6b`                                                                                                                                                           |
| Generation timed out              | Raise `SUMMARY_TIMEOUT_S` (default 240s) for slow models                                                                                                                               |
| API key missing                   | Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`                                                                                                                                            |
| `generation_status` is `pending`  | Wait for the entry's generation job; see [Deferred Summaries](#deferred-summaries)                                                                                                     |
| `generation_status` is `failed`   | Fix the provider, then restart the server or run `mcp-context-server-migrate --embed-missing`; see [Monitoring Deferred Generation](semantic-search.md#monitoring-deferred-generation) |

### Server Won't Start

**Error**: `ENABLE_SUMMARY_GENERATION=true but langchain-ollama package not installed`

**Solution**: Install the required extra:
```bash
uv sync --extra summary-ollama
```
Or disable summary generation: `ENABLE_SUMMARY_GENERATION=false`

### Poor Summary Quality

**Solutions:**
- Upgrade to a larger model (`qwen3:1.7b` or `qwen3:8b`)
- Increase `SUMMARY_MAX_TOKENS` to allow longer, more detailed summaries
- Provide a custom `SUMMARY_PROMPT` tailored to your domain

### Timeout Errors

**Error**: `Summary generation timed out after 240s`

**Solutions:**
- Increase `SUMMARY_TIMEOUT_S` (e.g., `300`)
- Use a smaller/faster model (`qwen3:0.6b`) or upgrade to `qwen3:1.7b` for better quality
- Reduce `SUMMARY_MAX_CONCURRENT` to limit parallel load on the model server (this single budget caps both flat document summaries and index_tree per-node summaries)
- A model that echoes a repetitive document until `SUMMARY_MAX_TOKENS` runs out times out on every attempt; a `qwen3` model gets the presence penalty that stops it, and another family needs `SUMMARY_OLLAMA_PRESENCE_PENALTY` (see [Reasoning Off](#reasoning-off))

## Additional Resources

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation including `summary` field
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - SUMMARY_EXTRA build argument
- **Main Documentation**: [README.md](../README.md) - overview and quick start
