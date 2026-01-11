# Compiler Tool

The compiler/distillation tool transforms aggregated research submissions into a coherent academic paper.

## Purpose

The compiler tool reads the aggregator's shared training database and systematically builds a complete academic paper through sequential validation cycles.

## Implementation Status

✅ **Fully Implemented**

## Features

- **Sequential Markov Chain Workflow**: One submitter runs at a time, each submission must be validated before proceeding
- **Multiple Submitter Modes**:
  - **Outline Creation/Update**: High-context model creates and maintains paper structure
  - **Paper Construction**: High-context model writes paper sections following the outline
  - **Review/Cleanup**: High-context model reviews and fixes errors (without aggregator DB context)
  - **Rigor Enhancement**: High-parameter model adds scientific rigor and precision
- **Real-time Paper Viewing**: Live updates in the GUI as the paper is constructed
- **Intelligent Placement Logic**: Automatically inserts content at the correct location based on placement context
- **Separate GUI Tabs**: Compiler Interface, Settings, Logs, and Live Paper view

## Architecture

### Core Components

- `compiler_coordinator.py` - Main orchestrator for sequential workflow
- `compiler_rag_manager.py` - Manages RAG retrieval for aggregator database

### Agents

- `high_context_submitter.py` - Low-parameter, high-context model (outline, construction, review)
- `high_param_submitter.py` - High-parameter, low-context model (rigor enhancement)

### Validation

- `compiler_validator.py` - Validates submissions for coherence, rigor, placement, and non-redundancy

### Memory

- `outline_memory.py` - Manages current paper outline
- `paper_memory.py` - Manages current paper state
- `compiler_rejection_log.py` - Tracks last 10 rejections and acceptances

## System-Managed Markers - Critical Architecture

The compiler uses two categories of hard-coded markers:

### 1. Section Placeholders (in paper during construction)

**Markers:**
- `[HARD CODED PLACEHOLDER FOR THE ABSTRACT SECTION - TO BE WRITTEN AFTER THE INTRODUCTION IS COMPLETE]`
- `[HARD CODED PLACEHOLDER FOR INTRODUCTION SECTION - TO BE WRITTEN AFTER THE CONCLUSION SECTION IS COMPLETE]`
- `[HARD CODED PLACEHOLDER FOR THE CONCLUSION SECTION - TO BE WRITTEN AFTER THE BODY SECTION IS COMPLETE]`

**Management:**
- Added by `paper_memory.py` via `initialize_with_placeholders()`
- Replaced by `paper_memory.replace_placeholder()` when sections are validated
- Purpose: Make it crystal clear to AI what sections exist vs. don't exist

### 2. Anchors (in paper and outline)

**Markers:**
- Paper: `[HARD CODED END-OF-PAPER MARK -- ALL CONTENT SHOULD BE ABOVE THIS LINE]`
- Outline: Two-line system (end-of-paper reference + end-of-outline mark)

**Management:**
- Added/maintained by `paper_memory.py` and `outline_memory.py`
- Purpose: Non-chronological stop tokens preventing content after endpoint

### Critical Distinction for Prompts

**Markers in CURRENT DOCUMENT/OUTLINE:**
- System-managed, expected, normal during construction
- NOT AI-generated content
- Code automatically adds and removes them

**Markers in SUBMISSION CONTENT:**
- Forbidden, invalid, auto-rejected by pre-validation
- Validator criterion #11 (NO PLACEHOLDER TEXT) checks submissions only
- Pre-validation check at `compiler_validator.py` line 326 catches this before validation

### Why Placeholders Fix Bugs

Previous system had AI falsely claiming sections were "already written" when they weren't. Placeholders make document state explicit:
- If placeholder present → section does NOT exist yet
- If actual content present → section IS written
- Eliminates AI confusion about document state

## Startup

The compiler starts manually via API only (`POST /api/compiler/start`). There is no automatic startup trigger.

## Configuration

Default context window: 131072 tokens (configurable in GUI settings)

## Integration

The compiler continuously reads from the aggregator's shared training database (`backend/data/rag_shared_training.txt`) and re-RAGs every 10 new aggregator acceptances.

