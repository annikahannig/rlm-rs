# FIX_RLM_RS_PLAN_V2.md - Complete Overhaul

## Root Cause Analysis

After re-reading the paper carefully, I found the **fundamental problem**: our implementation's system prompt is way too simple compared to the paper's actual system prompt (see Appendix D, pages 24-26).

The paper's system prompt is **~2500 words** with:
- Detailed context metadata
- Multiple code examples (5+ complete examples)
- Explicit chunking strategies
- Clear instructions about llm_query usage
- Step-by-step guidance

Our current prompt is **~200 words** with minimal guidance.

---

## Issue 1: System Prompt is Too Simple (CRITICAL)

**Problem**: The paper's system prompt (Appendix D.1, page 24-26) includes:

```
You are tasked with answering a query with associated context. You can access, transform, and analyze
this context interactively in a REPL environment that can recursively query sub-LLMs...

Your context is a {context_type} with {context_total_length} total characters, and is broken up into
chunks of char lengths: {context_lengths}.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query...
2. A 'llm_query' function that allows you to query an LLM...
3. The ability to use 'print()' statements...
```

Then it provides **5+ complete code examples** showing:
- How to chunk context
- How to use llm_query
- How to iterate through sections
- How to aggregate results
- How to use buffers

Our current prompt has NONE of this.

**Fix**: Completely rewrite `build_system_prompt()` to match the paper's structure.

---

## Issue 2: Missing Context Metadata

**Problem**: The paper provides detailed context metadata:
```
Your context is a {context_type} with {context_total_length} total characters,
and is broken up into chunks of char lengths: {context_lengths}.
```

We only pass `context_len`.

**Fix**: Add `context_type` parameter (always "string" for now).

---

## Issue 3: Missing FINAL/FINAL_VAR Instructions

**Problem**: The paper explicitly tells the model:
```
IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL
function when you have completed your task, NOT in code. Do not use these tags unless you have
completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment
```

Our prompt doesn't explain this clearly.

---

## Issue 4: Weak Model Compatibility

**Problem**: From Appendix A (page 13):
> "Models without sufficient coding capabilities struggle as RLMs. Our instantiation of RLMs relies
> on the ability to reason through and deal with the context in a REPL environment."

The user is testing with `cogito:14b` which may not have sufficient coding abilities for RLM.

**Note**: This might be the real issue - the model simply isn't capable enough.

---

## Issue 5: No Code Examples in Prompt

**Problem**: The paper's prompt includes 5+ complete Python code examples showing exactly how to:
1. Peek at context: `chunk = context[:10000]; print(f"First 10000 characters: {chunk}")`
2. Use llm_query: `answer = llm_query(f"What is the magic number? {chunk}")`
3. Iterate through chunks with buffers
4. Aggregate answers from multiple llm_query calls

**Fix**: Add multiple code examples to the system prompt.

---

## Implementation Plan

### Step 1: Rewrite System Prompt (Major Change)

Replace `src/prompts.rs` with the paper's actual prompt structure:

```rust
pub fn build_system_prompt(context_len: usize) -> String {
    format!(r#"You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a string with {context_len} total characters.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:

```repl
chunk = context[:1000]
print(f"First 1000 characters: {{chunk}}")
```

Example - using llm_query to analyze content:
```repl
chunk = context[:5000]
answer = llm_query(f"What task is the user asking for? Here is their message: {{chunk}}")
print(answer)
```

Example - for simple tasks, just read and do:
```repl
print(context)  # See what the user wants
```
Then if they want fibonacci:
```repl
fib = [1, 1]
for _ in range(20): fib.append(fib[-1] + fib[-2])
result = fib[:23]
print(result)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response. Output to the REPL environment as much as possible. Remember to explicitly answer the original query in your final answer."#, context_len = context_len)
}
```

### Step 2: Simplify Initial User Prompt

```rust
pub fn build_initial_user_prompt() -> String {
    "You have not interacted with the REPL environment yet. Start by examining the 'context' variable to understand your task. Your next action:".to_string()
}
```

### Step 3: Add Continue Prompt Back

```rust
pub fn build_continue_prompt() -> String {
    "Continue working on the task. If you have the answer, respond with FINAL(answer) or FINAL_VAR(variable_name). Your next action:".to_string()
}
```

### Step 4: Use Continue Prompt in rlm.rs

When no code blocks are produced, use the continue prompt instead of the nudge.

---

## Testing Recommendations

1. **Test with a stronger model first** - Try with a model that has good coding abilities:
   - GPT-4/GPT-4o via OpenAI API
   - Claude via Anthropic API
   - Qwen3-Coder (what the paper used)
   - DeepSeek-Coder

2. **Test with simple queries**:
   - "What is 2 + 2?"
   - "Generate fibonacci numbers 1-10"
   - "Count the words in the context"

3. **Enable verbose mode** to see exactly what's happening.

---

## Summary of Changes

| File | Change |
|------|--------|
| `src/prompts.rs` | Complete rewrite with paper's prompt structure |
| `src/rlm.rs` | Use continue prompt, simplify history flow |

---

## Model Requirements Note

From the paper (Appendix A, page 13):
> "Models without sufficient coding capabilities struggle as RLMs."

The paper tested with:
- GPT-5 (frontier model)
- Qwen3-Coder-480B-A35B (frontier open model)

Testing with `cogito:14b` may simply not work because the model lacks the coding reasoning abilities needed for RLM. Consider testing with a stronger model first to validate the implementation, then optimize prompts for weaker models if needed.
