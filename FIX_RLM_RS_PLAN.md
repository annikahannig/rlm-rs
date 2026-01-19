# FIX_RLM_RS_PLAN.md

## Problem Summary

The rlm-rs implementation has several issues causing the model to "gonk out" (hallucinate, lose context, re-execute code repeatedly). After reviewing the original RLM paper and comparing with the current implementation, I've identified the following issues:

---

## Issue 1: Duplicate History Entry (Critical Bug)

**Location:** `src/rlm.rs` lines 263 and 368

**Problem:** The assistant response is added to history **twice**:
- Line 263: `history.push(Message::assistant(&response_text));` (correct - before code execution)
- Line 368: `history.push(Message::assistant(&response_text));` (BUG - duplicate!)

This causes the model to see duplicate messages in its history, confusing it about what it has already said.

**Fix:** Remove the duplicate at line 368.

---

## Issue 2: Missing Context Metadata in System Prompt

**Location:** `src/prompts.rs`

**Problem:** The paper's system prompt includes crucial metadata:
```
Your context is a {context_type} with {context_total_length} total characters,
and is broken up into chunks of char lengths: {context_lengths}.
```

The current implementation doesn't provide this metadata. The model needs to know:
- The type of context (string, list, etc.)
- Total character length
- How the context is structured

**Fix:** Update `build_system_prompt()` to accept context metadata parameters and include them in the prompt.

---

## Issue 3: Initial User Prompt Should NOT Reveal Context Content

**Location:** `src/prompts.rs` - `build_initial_user_prompt()`

**Problem:** The paper's approach is that the model should **examine the context variable** to discover its task. The current implementation mentions the context length, but the model should be instructed to start by examining `context` without any preview.

**Fix:** Simplify the initial user prompt to just instruct the model to start by examining the `context` variable.

---

## Issue 4: Code Execution Results Format

**Location:** `src/rlm.rs` - code execution result handling

**Problem:** The current format for feeding code execution results back is verbose. The paper shows simpler output patterns. The model should see:
1. The code it executed
2. The output (stdout)
3. Any errors

**Fix:** Simplify the execution result format to be more like a REPL transcript.

---

## Issue 5: FINAL() Detection Should Handle In-Code Calls

**Location:** `src/parsing.rs` and `src/env/callback.rs`

**Problem:** The paper distinguishes between:
1. `FINAL(answer)` in the model's **text response** (outside code)
2. `FINAL(variable)` called **within code** which should print `FINAL_ANSWER: {value}`

Current implementation has both, but the code-based FINAL may not be working correctly.

**Fix:** Verify the FINAL() function in the REPL namespace properly captures and outputs the final answer.

---

## Implementation Plan

### Step 1: Fix Duplicate History Entry
```rust
// In src/rlm.rs, around line 366-380
// DELETE these lines (the duplicate history update):
// history.push(Message::assistant(&response_text));
```

### Step 2: Add Context Metadata to System Prompt
```rust
// In src/prompts.rs, change build_system_prompt() signature:
pub fn build_system_prompt(context_type: &str, context_len: usize) -> String {
    format!(r#"You are tasked with answering a query with associated context...

Your context is a {context_type} with {context_len} total characters.

The REPL environment is initialized with:
..."#, context_type = context_type, context_len = context_len)
}
```

### Step 3: Simplify Initial User Prompt
```rust
// In src/prompts.rs
pub fn build_initial_user_prompt() -> String {
    "You have not interacted with the REPL environment yet. \
     Start by examining the `context` variable to understand your task. \
     Your next action:".to_string()
}
```

### Step 4: Simplify Continue Prompt
```rust
pub fn build_continue_prompt() -> String {
    "Continue examining the context and using the REPL. \
     Use FINAL() or FINAL_VAR() when you have your answer. \
     Your next action:".to_string()
}
```

### Step 5: Fix Code Execution Result Format
```rust
// Simpler format that looks like actual REPL output
fn format_execution_result(result: &ReplResult) -> String {
    let mut output = String::new();

    if !result.stdout.is_empty() {
        output.push_str(&result.stdout);
    }

    if !result.stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&result.stderr);
    }

    if output.is_empty() {
        "(no output)".to_string()
    } else {
        output
    }
}
```

### Step 6: Update History Flow
The correct flow should be:
1. Build messages for LLM call (system + history)
2. Call LLM, get response
3. Add assistant response to history
4. Extract and execute code blocks
5. For each code block, add execution result as user message
6. If no code blocks and no final answer, add continue prompt
7. Loop

---

## Files to Modify

1. **`src/rlm.rs`**
   - Remove duplicate `history.push(Message::assistant(&response_text))` at line 368
   - Update call to `build_system_prompt()` with metadata
   - Simplify `format_execution_result()`

2. **`src/prompts.rs`**
   - Add context metadata to system prompt
   - Simplify initial and continue prompts
   - Match paper's prompt structure more closely

3. **`src/env/pyo3_repl.rs`** (if needed)
   - Verify FINAL() function works correctly in REPL namespace

4. **`crates/rlm_chat/src/main.rs`**
   - Ensure context is built correctly for chat use case

---

## Testing Plan

1. Run with verbose mode to verify:
   - History contains no duplicate messages
   - Context metadata appears in system prompt
   - Code execution results are clean
   - FINAL() detection works from both text and code

2. Test with simple prompts:
   - `"What is 2 + 2?"`
   - `"Generate fibonacci sequence 1..10"`
   - Multi-step reasoning tasks

3. Verify the model:
   - Correctly examines `context` variable first
   - Doesn't hallucinate about context content
   - Progresses through iterations without repeating
   - Terminates with FINAL() when done

---

## Key Insight from Paper

The core RLM insight is:
> "Long prompts should not be fed into the neural network directly but should instead be treated as part of the environment that the LLM can symbolically interact with."

The model should:
1. Receive a **short system prompt** with instructions and context metadata
2. Be told the `context` variable exists with N characters
3. Use **code execution** to examine, chunk, and process the context
4. Use **llm_query()** for recursive sub-calls on chunks
5. Build up answers in **REPL variables**
6. Return final answer via **FINAL()** or **FINAL_VAR()**

The model should NOT see the full context in its token window - it should only see the context through REPL print statements (which are truncated).
