# Development rules

## Before editing
- Never assume API parameters, formats, or versions:
    - Check `./docs_and_repos/` for API references; use git clone/pull and web search if required
    - Get current library documentation and examples from Context7 (MCP)
    - Query official repositories for implementation details (DeepWiki MCP)
    - Verify latest API versions and parameters (Web search/ Github MCP)
- Verify your TODO/plans against existing requirements and architecture design documented in CLAUDE.md

## After editing
- Verify edits against existing requirements and architecture design
- Run lint: `./lint.sh` - Fix all errors/ warnings, check again
- Refactor if complexity limits exceeded
- Verify the codebase does not contain:
    - silent error/warning drops
    - exception silencing
    - error/fatal logging without raising an exception
    - monkey patching
    - modifying/ commenting out broken code
    - modifying/ commenting out tests

## Code quality standards (`./lint.sh`)

### Python
- pylint - 10.0/10
- basedpyright - 0 errors, 0 warnings
- ruff - No errors/warnings

### Shell scripts
- POSIX shell compatibility
- shellcheck - No warnings


# Guidelines
- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files.


<!-- sep---sep -->


# Architecture

## OpenAI chat/completion <-> responses bridge
```
Client → LiteLLM → OpenAI Responses Bridge → OpenAI Python Lib → OpenAI Responses API
(chat streaming) → (stream simulation + prefix + heartbeat) → (responses non-streaming + tools)
```


## Perplexity proxy bridge
```
Client → LiteLLM → Perplexity Bridge → Perplexity API
(chat/completion only) → (direct passthrough + search_results formatting)
```

