use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// Pre-compiled regexes for performance
static CODE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"```(?:repl|python)\n([\s\S]*?)```").expect("invalid regex")
});

/// Extract code blocks delimited by ```repl``` or ```python``` markers
pub fn extract_code_blocks(text: &str) -> Vec<String> {
    CODE_BLOCK_RE
        .captures_iter(text)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect()
}

/// Check for FINAL(answer) pattern - handles nested parentheses correctly
pub fn extract_final_answer(text: &str) -> Option<String> {
    extract_final_answer_raw(text, &HashMap::new())
}

/// Check for FINAL(answer) pattern with variable resolution from locals
///
/// This function is strict about what constitutes a valid FINAL() call:
/// - Must be at the start of a line OR preceded by whitespace/punctuation
/// - Content must not look like English prose (descriptive text)
pub fn extract_final_answer_raw(text: &str, locals: &HashMap<String, String>) -> Option<String> {
    let start_marker = "FINAL(";

    // Find all occurrences and check each one
    let mut search_start = 0;
    while let Some(pos) = text[search_start..].find(start_marker) {
        let start_pos = search_start + pos;

        // Check that FINAL( is at a valid position:
        // - Start of string
        // - After newline
        // - After whitespace
        // - After colon (like "Answer: FINAL()")
        let valid_position = start_pos == 0 || {
            let prev_char = text[..start_pos].chars().last().unwrap();
            prev_char == '\n' || prev_char.is_whitespace() || prev_char == ':'
        };

        if !valid_position {
            search_start = start_pos + 1;
            continue;
        }

        let content_start = start_pos + start_marker.len();

        // Count parentheses to find the matching close
        // - Skip parens inside quoted strings (handles unbalanced parens in content)
        // - Skip emoticon patterns like :) ;) =)
        let mut depth = 1;
        let mut end_pos = None;
        let remaining = &text[content_start..];
        let mut in_string: Option<char> = None; // Track if we're inside a string and which quote

        for (i, ch) in remaining.char_indices() {
            // Handle string tracking (skip parens inside strings)
            if in_string.is_some() {
                if Some(ch) == in_string {
                    // Check for escape (look at previous char)
                    let is_escaped = i > 0 && remaining[..i].ends_with('\\');
                    if !is_escaped {
                        in_string = None;
                    }
                }
                continue; // Don't count parens inside strings
            }

            // Check for string start
            if ch == '"' || ch == '\'' {
                in_string = Some(ch);
                continue;
            }

            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        // Check if this looks like an emoticon (skip it if so)
                        let is_emoticon = i > 0 && {
                            let prev_char = remaining[..i].chars().last().unwrap();
                            matches!(prev_char, ':' | ';' | '=' | '8' | 'X' | 'x' | 'D' | 'P' | 'p')
                        };

                        if is_emoticon {
                            // This is likely an emoticon, keep going
                            depth = 1; // Reset depth to continue looking
                        } else {
                            end_pos = Some(content_start + i);
                            break;
                        }
                    }
                }
                _ => {}
            }
        }

        // If we didn't find a valid close, try again without emoticon skipping
        if end_pos.is_none() {
            depth = 1;
            in_string = None;
            for (i, ch) in remaining.char_indices() {
                // Handle string tracking
                if in_string.is_some() {
                    if Some(ch) == in_string {
                        let is_escaped = i > 0 && remaining[..i].ends_with('\\');
                        if !is_escaped {
                            in_string = None;
                        }
                    }
                    continue;
                }
                if ch == '"' || ch == '\'' {
                    in_string = Some(ch);
                    continue;
                }

                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end_pos = Some(content_start + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(end) = end_pos {
            let content = text[content_start..end].trim().to_string();

            // Reject if content looks like descriptive English text
            // (e.g., "Output from executing code", "The result of the calculation")
            if looks_like_prose(&content) {
                search_start = end + 1;
                continue;
            }

            // If content looks like a variable name (identifier), try to resolve it from locals
            if is_identifier(&content) {
                if let Some(value) = locals.get(&content) {
                    return Some(value.clone());
                }
            }

            // If content is a quoted string, unescape it
            let content = unescape_string_literal(&content);

            return Some(content);
        }

        search_start = start_pos + 1;
    }

    None
}

/// Check if text looks like descriptive English prose rather than an answer
fn looks_like_prose(text: &str) -> bool {
    let lower = text.to_lowercase();

    // If text contains code-like patterns (function calls), it's probably valid
    // even if it happens to contain some English words
    if has_code_patterns(text) {
        return false;
    }

    // Common prose patterns that indicate descriptive text, not actual answers
    // These must be at the START of the content to be considered prose
    let prose_prefixes = [
        "output from",
        "result of",
        "this is the",
        "this is a",
        "the result",
        "here is",
    ];

    for prefix in &prose_prefixes {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    // Words that strongly indicate meta-description anywhere in text
    let strong_prose_indicators = [
        "executing code",
        "execution of",
        "demonstration of",
        "example of how",
    ];

    for indicator in &strong_prose_indicators {
        if lower.contains(indicator) {
            return true;
        }
    }

    false
}

/// Check if text contains code-like patterns (function calls, math operations)
fn has_code_patterns(text: &str) -> bool {
    // Look for function call patterns like foo(x) or bar(1, 2)
    let mut chars = text.chars().peekable();
    let mut in_identifier = false;

    while let Some(c) = chars.next() {
        if c.is_alphabetic() || c == '_' {
            in_identifier = true;
        } else if c == '(' && in_identifier {
            // Found identifier followed by ( - looks like function call
            return true;
        } else if !c.is_alphanumeric() && c != '_' {
            in_identifier = false;
        }
    }

    // Also check for math/code operators
    text.contains('+') || text.contains('*') || text.contains('/') || text.contains('[')
}

/// Check if a string is a valid Python identifier (variable name)
fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Strip quotes and unescape \n to actual newlines
fn unescape_string_literal(s: &str) -> String {
    let t = s.trim();
    // Strip matching quotes
    let inner = if (t.starts_with('"') && t.ends_with('"'))
        || (t.starts_with('\'') && t.ends_with('\''))
    {
        &t[1..t.len() - 1]
    } else {
        return s.to_string();
    };
    // Simple unescape
    inner.replace("\\n", "\n").replace("\\t", "\t")
}

/// Extract FINAL answer - handles both FINAL("literal") and FINAL(var)
pub fn extract_answer(text: &str, locals: &HashMap<String, String>) -> Option<String> {
    extract_final_answer_raw(text, locals)
}

/// Extract FINAL_ANSWER from code execution stdout
/// This is printed when FINAL() is called from within code
pub fn extract_final_answer_from_stdout(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        if let Some(answer) = line.strip_prefix("FINAL_ANSWER: ") {
            return Some(answer.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_blocks_repl() {
        let text = r#"
Here's some code:

```repl
x = 1 + 1
print(x)
```

And more text.
"#;
        let blocks = extract_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0], "x = 1 + 1\nprint(x)\n");
    }

    #[test]
    fn test_extract_code_blocks_python() {
        let text = r#"
```python
def foo():
    return 42
```
"#;
        let blocks = extract_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("def foo():"));
    }

    #[test]
    fn test_extract_multiple_code_blocks() {
        let text = r#"
First block:
```repl
a = 1
```

Second block:
```python
b = 2
```
"#;
        let blocks = extract_code_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("a = 1"));
        assert!(blocks[1].contains("b = 2"));
    }

    #[test]
    fn test_extract_code_blocks_none() {
        let text = "No code blocks here, just plain text.";
        let blocks = extract_code_blocks(text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_extract_code_blocks_other_language() {
        let text = r#"
```javascript
console.log("hello");
```
"#;
        let blocks = extract_code_blocks(text);
        assert!(blocks.is_empty()); // Only repl/python
    }

    #[test]
    fn test_extract_final_answer_simple() {
        let text = "The answer is FINAL(42)";
        assert_eq!(extract_final_answer(text), Some("42".to_string()));
    }

    #[test]
    fn test_extract_final_answer_with_text() {
        let text = "After calculation, FINAL(hello world) is the result.";
        assert_eq!(extract_final_answer(text), Some("hello world".to_string()));
    }

    #[test]
    fn test_extract_final_answer_multiline() {
        let text = r#"FINAL(line 1
line 2
line 3)"#;
        let answer = extract_final_answer(text).unwrap();
        assert!(answer.contains("line 1"));
        assert!(answer.contains("line 2"));
        assert!(answer.contains("line 3"));
    }

    #[test]
    fn test_extract_final_answer_none() {
        let text = "No final answer here";
        assert_eq!(extract_final_answer(text), None);
    }

    #[test]
    fn test_extract_final_answer_nested_parens() {
        let text = "FINAL(The answer is foo(x) + bar(y, z))";
        assert_eq!(
            extract_final_answer(text),
            Some("The answer is foo(x) + bar(y, z)".to_string())
        );
    }

    #[test]
    fn test_extract_final_answer_deeply_nested() {
        let text = "FINAL(outer(inner(deep(value))))";
        assert_eq!(
            extract_final_answer(text),
            Some("outer(inner(deep(value)))".to_string())
        );
    }

    #[test]
    fn test_extract_final_resolves_variable() {
        // FINAL(var) looks up variable in locals
        let text = "The result is FINAL(result)";
        let mut locals = HashMap::new();
        locals.insert("result".to_string(), "computed_value".to_string());

        assert_eq!(
            extract_answer(text, &locals),
            Some("computed_value".to_string())
        );
    }

    #[test]
    fn test_extract_final_variable_not_found_returns_name() {
        // If variable not in locals, return the identifier as-is
        let text = "FINAL(missing)";
        let locals = HashMap::new();
        assert_eq!(extract_answer(text, &locals), Some("missing".to_string()));
    }

    #[test]
    fn test_extract_final_literal_string() {
        // FINAL("literal") strips quotes
        let text = r#"FINAL("hello world")"#;
        let locals = HashMap::new();
        assert_eq!(extract_answer(text, &locals), Some("hello world".to_string()));
    }

    #[test]
    fn test_extract_final_literal_with_newlines() {
        // FINAL("foo\nbar") unescapes to actual newlines
        let text = r#"FINAL("line1\nline2\nline3")"#;
        let locals = HashMap::new();
        assert_eq!(extract_answer(text, &locals), Some("line1\nline2\nline3".to_string()));
    }

    #[test]
    fn test_extract_final_rejects_prose_output() {
        // This was a real failure case - model said "FINAL(Output from executing code)"
        let text = "Here's the FINAL(Output from executing code) result.";
        assert_eq!(extract_final_answer(text), None);
    }

    #[test]
    fn test_extract_final_rejects_prose_result_of() {
        let text = "FINAL(the result of the calculation)";
        assert_eq!(extract_final_answer(text), None);
    }

    #[test]
    fn test_extract_final_rejects_prose_demonstration() {
        let text = "FINAL(This is a demonstration of the system)";
        assert_eq!(extract_final_answer(text), None);
    }

    #[test]
    fn test_extract_final_accepts_valid_at_line_start() {
        let text = "FINAL(42)";
        assert_eq!(extract_final_answer(text), Some("42".to_string()));
    }

    #[test]
    fn test_extract_final_accepts_after_newline() {
        let text = "Some text\nFINAL(the answer)";
        assert_eq!(extract_final_answer(text), Some("the answer".to_string()));
    }

    #[test]
    fn test_extract_final_accepts_after_colon() {
        let text = "Answer: FINAL(123)";
        assert_eq!(extract_final_answer(text), Some("123".to_string()));
    }

    #[test]
    fn test_extract_final_accepts_numbers_list() {
        let text = "FINAL(1, 1, 2, 3, 5, 8, 13, 21)";
        assert_eq!(extract_final_answer(text), Some("1, 1, 2, 3, 5, 8, 13, 21".to_string()));
    }

    #[test]
    fn test_extract_final_skips_prose_finds_valid() {
        // First FINAL is prose, second is valid
        let text = "FINAL(Output from executing code)\nFINAL(42)";
        assert_eq!(extract_final_answer(text), Some("42".to_string()));
    }

    #[test]
    fn test_extract_final_skips_smiley_emoticon() {
        // Smiley face :) should not close FINAL
        let text = "FINAL(answer :) here)";
        assert_eq!(extract_final_answer(text), Some("answer :) here".to_string()));
    }

    #[test]
    fn test_extract_final_skips_wink_emoticon() {
        // Wink ;) should not close FINAL
        let text = "FINAL(great job ;) done)";
        assert_eq!(extract_final_answer(text), Some("great job ;) done".to_string()));
    }

    #[test]
    fn test_extract_final_normal_parens_still_work() {
        // Normal parens should still close at first depth=0
        let text = "FINAL(42) is the answer";
        assert_eq!(extract_final_answer(text), Some("42".to_string()));
    }

    #[test]
    fn test_extract_final_nested_with_emoticon() {
        // Nested parens followed by emoticon
        let text = "FINAL(foo(bar) and :) end)";
        assert_eq!(extract_final_answer(text), Some("foo(bar) and :) end".to_string()));
    }

    #[test]
    fn test_extract_final_quoted_string_with_unbalanced_parens() {
        // Quoted string content with unbalanced parens (like ASCII art)
        let text = r#"FINAL(
"Here's some art with unbalanced parens:
.(.
(..)
Pretty cool!"
)"#;
        let answer = extract_final_answer(text).unwrap();
        assert!(answer.contains("unbalanced parens"));
        assert!(answer.contains(".(."));
    }

    #[test]
    fn test_extract_final_multiline_quoted_content() {
        // Real-world case: multi-line FINAL with quoted string
        let text = r#"FINAL(
"The answer is (1+2) = 3.
Here's more: ((.)) art"
)"#;
        let answer = extract_final_answer(text).unwrap();
        assert!(answer.contains("The answer is"));
        assert!(answer.contains("(1+2)"));
    }
}
