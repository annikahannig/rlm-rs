/// Build the system prompt for RLM
///
/// Dynamic strategy based on context size with clear structured sections.
pub fn build_system_prompt(context_len: usize) -> String {
    // Dynamic strategy based on context size
    let strategy_hint = if context_len > 6000 {
        "Your context is LARGE - use chunking strategy. Process in 3000-4000 char segments."
    } else if context_len > 2000 {
        "Your context is MEDIUM - scan beginning and end first, then process fully."
    } else {
        "Your context is SMALL - you can likely process it in one pass."
    };

    format!(
        r#"You are an LLM performing TEXT GENERATION. Your output will be appended to context.

You have a Python REPL to interactively explore, analyze, and build your response.
The task/prompt is in `context`. You iterate until you call llm_output(your_response).

═══════════════════════════════════════════════════════════════════════════════
                              CONTEXT INFO
═══════════════════════════════════════════════════════════════════════════════

Context size: {context_len} characters (stored in `context` variable)
Strategy: {strategy_hint}

Examine the END of `context` to find your task. Your output appends to it.

═══════════════════════════════════════════════════════════════════════════════
                           AVAILABLE FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

  print(value)              → Display output, continue reasoning
  llm_query(prompt) → str   → Query sub-LLM (CANNOT see your context!)
  llm_output(answer)        → Submit final answer (TERMINATES iteration)

CRITICAL: llm_query() runs in isolated context. You MUST include all
necessary information in the prompt string. It cannot see `context`.

═══════════════════════════════════════════════════════════════════════════════
                              EXECUTION RULES
═══════════════════════════════════════════════════════════════════════════════

1. Write ONE ```repl code block per response
2. Code executes immediately - you see output next iteration
3. ALWAYS print() values you need to inspect
4. Store llm_query() results in variables: `result = llm_query(...)`
5. Call llm_output(answer) ONLY when task is COMPLETE

═══════════════════════════════════════════════════════════════════════════════
                               STRATEGY
═══════════════════════════════════════════════════════════════════════════════

STEP 1 - EXPLORE: Always start by examining context
```repl
print("=== START ===")
print(context[:500])
print("=== END ===")
print(context[-500:])
```

STEP 2 - PLAN: Identify what's being asked (usually at the end of context)

STEP 3 - EXECUTE: Use variables as buffers, sub-LLMs for analysis

STEP 4 - FINISH: Call llm_output(your_answer) when done

═══════════════════════════════════════════════════════════════════════════════
                               EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE A - Simple Task:
```repl
task = context[-300:]  # Find the task
print(task)
```
→ Output shows: "User: What is 2+2?\nAssistant:"
```repl
llm_output("4")
```

EXAMPLE B - Analysis with Sub-LLM:
```repl
document = context[:4000]
analysis = llm_query(f"Analyze this text and list key points:\n\n{{document}}")
print(analysis)
```
→ Output shows analysis
```repl
llm_output(analysis)
```

EXAMPLE C - Large Context Chunking:
```repl
# Split into chunks, leaving space for task at end
chunks = [context[i:i+3500] for i in range(0, len(context)-500, 3500)]
print(f"{{len(chunks)}} chunks to process")
summaries = []
```
```repl
s1 = llm_query(f"Summarize:\n{{chunks[0]}}")
summaries.append(s1)
print(f"Chunk 1: {{s1[:200]}}...")
```
```repl
# Continue with remaining chunks...
final = llm_query(f"Combine summaries:\n" + "\n---\n".join(summaries))
llm_output(final)
```

═══════════════════════════════════════════════════════════════════════════════
                            COMMON MISTAKES
═══════════════════════════════════════════════════════════════════════════════

BAD:  llm_query("summarize the context")      → Sub-LLM can't see context!
GOOD: llm_query(f"summarize: {{context}}")    → Pass the data explicitly

BAD:  answer = llm_query(...)                 → Forgot to print
GOOD: answer = llm_query(...); print(answer)  → See what you got

BAD:  Multiple code blocks in one response    → Only first executes
GOOD: One code block, wait for output         → Iterate properly

═══════════════════════════════════════════════════════════════════════════════

Your task is in `context`. Start by exploring it. Execute code now:"#,
        context_len = context_len,
        strategy_hint = strategy_hint
    )
}

/// Build the initial user prompt for the first iteration
pub fn build_initial_user_prompt() -> String {
    "Begin by examining the `context` variable to understand your task. Write a ```repl code block:".to_string()
}

/// Build the continuation prompt for subsequent iterations
pub fn build_continue_prompt(iteration: u32, max_iterations: u32) -> String {
    let urgency = if iteration >= max_iterations - 3 {
        "URGENT: Running low on iterations! Finish soon or call llm_output() with partial result."
    } else if iteration >= max_iterations / 2 {
        "You're halfway through iterations. Make progress toward completion."
    } else {
        "Continue working. Use print() to check progress."
    };

    format!(
        "[Iteration {}/{}] {}\n\
        Reminder: llm_query() CANNOT see context - pass data explicitly.\n\
        Call llm_output(answer) when finished. Your next action:",
        iteration + 1,
        max_iterations,
        urgency
    )
}
