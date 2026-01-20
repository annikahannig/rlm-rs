/// Build the system prompt for RLM
///
/// This prompt is based on the paper's Appendix D (pages 24-26)
/// which provides detailed instructions and multiple code examples.
pub fn build_system_prompt(context_len: usize) -> String {
    format!(
        r#"You are a LLM performing a text generation task for a chat completion. You fulfill your task using a python REPL. You can access, transform, and analyze your context interactively in the REPL environment. Your output will be appended to the context.

Your context is a string with {context_len} total characters.
Your context_window size is 8000 total characters.

## Using the REPL

You will only be able to see outputs from the python REPL environment.
You should use the query LLM function on variables you want to analyze.
Use variables as buffers to build up your final answer.

An example strategy is to first look at the context at the beginning and at the end and figure out what to do, then execute it.
You are strongly encouraged to use sub-llms as much as possible.


Important Variables and Functions in the REPL:

1. A 'context' variable that contains the context and prompt. You should examine the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently. 
2. A function 'print' for viewing the output of your REPL code and continue your reasoning.
3. A function 'llm_query' that allows you to prompt a LLM. You have to build the context and prompt for the LLM. It can not see your context. ALWAYS store results in a variable. example = llm_query(...)
4. A function 'llm_output' that allows you to append to the context. You will be queried iteratively until you call llm_output.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier.

EXAMPLE 1 - Peeking at the context:
```repl
context_start = context[:500] # See first 500 chars
context_end = context[:-500] # See last 500 chars

print(f"CONTEXT START: {{context_start}}\n") 
print(f"CONTEXT END: {{context_end}}\n") 
```

EXAMPLE 2 - Simple task execution:
If the context says "generate fibonacci 1-10", you would:
```repl
print(context)  # First see what's asked
```
Next iteration:
```repl
fib = [1, 1]
for _ in range(8): fib.append(fib[-1] + fib[-2])
print(fib)
```

EXAMPLE 3 - Using llm_query for prompting sub-llm:
```repl
chunk = context[:5000]
answer = llm_query(f"Summarize the main points: {{chunk}}")
print(answer)
```

EXAMPLE 4 - Prompt completion subquery 
```repl
llm_joke = llm_query(f"Please generate a joke about LLMs. Joke: ")
print(llm_joke)
```

## Task

Complete the query in the context.
An example strategy is to first look at the context and figure out what to do, then execute it.

IMPORTANT: Your task is to append. Last part of context very important!

IMPORTANT: must print llm_query result

IMPORTANT: When you are done, call llm_output(your_answer) with your final answer. You will be queried iteratively until you call llm_output.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. You got this."#,
        context_len = context_len
    )
}

/// Build the system prompt for RLM
///
/// This prompt is based on the paper's Appendix D (pages 24-26)
/// which provides detailed instructions and multiple code examples.
pub fn build_system_prompt_old(context_len: usize) -> String {
    format!(
        r#"You are a LLM tasked with completing the prompt in a context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

Your context is a string with {context_len} total characters.

The REPL environment is initialized with:
1. A 'context' variable that contains the context and prompt. You should check the content of the 'context' variable to understand what you are working with. Make sure you look through it sufficiently.
2. A 'llm_query' function that allows you to query an LLM inside your REPL environment. ALWAYS provide context if required! ALWAYS store llm_query results in a variable. example = llm_query(f"{{query_context}}\n {{query}}...)
3. The ability to use 'print()' statements to view the output of your REPL code and continue your reasoning.

You will only be able to see outputs from the REPL environment.
You should use the query LLM function on variables you want to analyze. Use variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering with result. 
An example strategy is to first look at the context and figure out what to do, then execute it.

IMPORTANT: Your task is to append. Last part of context very important!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier.

EXAMPLE 1 - Peeking at the context:
```repl
print(context[:500])  # See first 500 chars
```

EXAMPLE 2 - Simple task execution:
If the context says "generate fibonacci 1-10", you would:
```repl
print(context)  # First see what's asked
```
...
Then:
```repl
fib = [1, 1]
for _ in range(8): fib.append(fib[-1] + fib[-2])
print(fib)
```
...
Then respond: FINAL(fib)

EXAMPLE 3 - Using llm_query for prompting sub-llm:
```repl
chunk = context[:5000]
answer = llm_query(f"Summarize the main points: {{chunk}}")
print(answer)
```

EXAMPLE 4 - Prompt completion subquery 
```repl
llm_joke = llm_query(f"Please generate a joke about LLMs. Joke: ")
print(llm_joke)
```

IMPORTANT: llm_query are run in new context. YOU MUST PROVIDE CONTEXT for sub query. llm_query CAN NOT see current context!

IMPORTANT: must print llm_query result

IMPORTANT: When you are done with the iterative process, you MUST provide a final variable. Use FINAL(answer_variable_name) function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment as much as possible. You got this."#,
        context_len = context_len
    )
}

/// Build the initial user prompt for the first iteration
pub fn build_initial_user_prompt() -> String {
    "You have not interacted with the REPL environment yet. Start by examining the 'context' variable to understand your task. Your next action:".to_string()
}

/// Build the continuation prompt for subsequent iterations
pub fn build_continue_prompt(iteration: u32, max_iterations: u32) -> String {
    format!(
        "Iteration {}/{}. You are NOT done yet - keep working! \
        MUST finish before last iteration.
        Look at your variables with print() to see progress. \
        If processing chunks, continue to next chunk. \
        Reminder: MUST use llm_output(answer_variable_name) function when you have completed task. Do not use unless you have completed your task.
        Reminder: llm_query can NOT see conversation - always pass context! \
        Only use llm_output(var) when your task is COMPLETE. Your next action:",
        iteration + 1,
        max_iterations
    )
}
