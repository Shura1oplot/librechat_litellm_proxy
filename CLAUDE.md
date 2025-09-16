# Proxy Architecture

## OpenAI Responses Bridge
```
Client → LiteLLM → OpenAI Responses Bridge → OpenAI Python Lib → OpenAI Responses API
(chat streaming) → (stream simulation + :test: + heartbeat) → (responses non-streaming + tools)
```

### Core Implementation
- `OpenAIResponsesBridge` class in `custom_handler.py`
- Direct OpenAI client calls (no monkey patching)
- Web search tools enabled by default
- Response prefix: `:test:` on all responses
- Heartbeat: `:hb:` every 5s (streaming only)
- Parameter forwarding: `reasoning_effort` with error handling

## Perplexity Proxy Bridge
```
Client → LiteLLM → Perplexity Bridge → Perplexity API
(chat/completion only) → (direct passthrough + search_results formatting)
```

### Core Implementation
- `PerplexityBridge` class in `custom_handler.py`
- Custom provider for chat/completion endpoints only
- Direct API calls to `https://api.perplexity.ai`
- **Non-streaming**: Add formatted `response.search_results` to response
- **Streaming**: Send `search_results` in final chunk as regular delta with proper `finish_reason`
- Parameter forwarding: All standard OpenAI + Perplexity-specific params
- Search result formatting: Clean presentation of sources with titles, URLs, dates


# Development Checklists

## ✅ BEFORE CODING CHECKLIST

### 🔍 Research Phase (MANDATORY)
- [ ] **Local docs first**: Check `./docs_and_repos/` for API references
- [ ] **Context7**: Get current library documentation and examples
- [ ] **Web search**: Verify latest API versions and parameters
- [ ] **DeepWiki**: Query official repositories for implementation details
- [ ] **NO ASSUMPTIONS**: Never assume API parameters, formats, or versions

### 📋 Planning Phase
- [ ] **Architecture check**: Verify plans against existing requirements
- [ ] **Zero tolerance**: All warnings = ERRORS (unless documented in CLAUDE.md)
- [ ] **No banned practices**: No silent drops, exception silencing, monkey patching

## ✅ AFTER EDITING CHECKLIST

### 🧪 Quality Validation (MANDATORY)
- [ ] **Architecture compliance**: Verify edits against existing requirements
- [ ] **Run lint**: `./lint.sh` - Fix ALL warnings/errors, check again
- [ ] **Complexity**: Refactor if complexity limits exceeded
- [ ] **Parameters**: Validate all forwarded correctly, no silent drops

### ⚡ Standards Verification
| Tool | Requirement | Status |
|------|------------|---------|
| pylint | 10.0/10 (ZERO warnings) | [ ] |
| basedpyright | 0 errors, 0 warnings | [ ] |
| ruff | No errors/warnings | [ ] |
| black | No formatting issues | [ ] |
| shellcheck | No warnings | [ ] |

### 🎯 Final Checks
- [ ] **Architecture compliance**: Matches documented patterns
- [ ] **Error handling**: No silent failures, clear error messages
- [ ] **Documentation**: Code changes align with CLAUDE.md constraints

---

# Core Standards

## Zero Tolerance Policy
- **Silent parameter dropping**: FORBIDDEN
- **Exception silencing** (`except Exception: pass`): BANNED
- **Monkey patching**: BANNED
- **Commenting out broken code**: BANNED
- **Logging without action**: BANNED

## Research Requirements
1. **Local docs**: `./docs_and_repos/` → 2. **Context7** → 3. **Web search** → 4. **DeepWiki** → 5. **Cross-reference** → 6. **Code**

## Quality Standards
- All warnings = ERRORS (unless documented)
- Function complexity ≤ 10 (STRICT)
- Max branches: 12 | Max statements: 50
- Parameter validation: explicit rejection over silent drops

# Important Instructions
- Do what has been asked; nothing more, nothing less.
- NEVER create files unless they're absolutely necessary for achieving your goal.
- ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if - explicitly requested by the User.
