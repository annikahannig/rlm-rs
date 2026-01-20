//! Built-in tools for the agent

use crate::{Tool, ToolResult};
use std::process::Command;

/// Echo tool - for testing
pub struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echo back the input (for testing)"
    }

    fn usage(&self) -> &str {
        "<tool:echo>message</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        ToolResult::ok(args)
    }
}

/// Read file tool
pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read contents of a file"
    }

    fn usage(&self) -> &str {
        "<tool:read_file>path/to/file.txt</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        let path = args.trim();
        match std::fs::read_to_string(path) {
            Ok(content) => ToolResult::ok(content),
            Err(e) => ToolResult::err(format!("Failed to read '{}': {}", path, e)),
        }
    }
}

/// Write file tool
pub struct WriteFileTool;

impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file. Format: path|||content"
    }

    fn usage(&self) -> &str {
        "<tool:write_file>path/to/file.txt|||file content here</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        let parts: Vec<&str> = args.splitn(2, "|||").collect();
        if parts.len() != 2 {
            return ToolResult::err("Invalid format. Use: path|||content");
        }

        let path = parts[0].trim();
        let content = parts[1];

        match std::fs::write(path, content) {
            Ok(()) => ToolResult::ok(format!("Written {} bytes to {}", content.len(), path)),
            Err(e) => ToolResult::err(format!("Failed to write '{}': {}", path, e)),
        }
    }
}

/// List directory tool
pub struct ListDirTool;

impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List files in a directory"
    }

    fn usage(&self) -> &str {
        "<tool:list_dir>path/to/directory</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        let path = args.trim();
        let path = if path.is_empty() { "." } else { path };

        match std::fs::read_dir(path) {
            Ok(entries) => {
                let mut files: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        if e.path().is_dir() {
                            format!("{}/", name)
                        } else {
                            name
                        }
                    })
                    .collect();
                files.sort();
                ToolResult::ok(files.join("\n"))
            }
            Err(e) => ToolResult::err(format!("Failed to list '{}': {}", path, e)),
        }
    }
}

/// Shell command tool (use with caution!)
pub struct ShellTool {
    pub allowed_commands: Vec<String>,
}

impl ShellTool {
    pub fn new() -> Self {
        Self {
            allowed_commands: vec![
                "ls".to_string(),
                "cat".to_string(),
                "head".to_string(),
                "tail".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "wc".to_string(),
                "date".to_string(),
                "pwd".to_string(),
                "echo".to_string(),
            ],
        }
    }

    pub fn allow_all() -> Self {
        Self {
            allowed_commands: vec![],
        }
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Run a shell command"
    }

    fn usage(&self) -> &str {
        "<tool:shell>ls -la</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        let cmd = args.trim();

        // Check if command is allowed
        if !self.allowed_commands.is_empty() {
            let first_word = cmd.split_whitespace().next().unwrap_or("");
            if !self.allowed_commands.iter().any(|c| c == first_word) {
                return ToolResult::err(format!(
                    "Command '{}' not allowed. Allowed: {:?}",
                    first_word, self.allowed_commands
                ));
            }
        }

        match Command::new("sh").arg("-c").arg(cmd).output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    ToolResult::ok(stdout.to_string())
                } else {
                    ToolResult::err(format!("Exit {}: {}", output.status, stderr))
                }
            }
            Err(e) => ToolResult::err(format!("Failed to run command: {}", e)),
        }
    }
}

/// Calculator tool
pub struct CalcTool;

impl Tool for CalcTool {
    fn name(&self) -> &str {
        "calc"
    }

    fn description(&self) -> &str {
        "Evaluate a mathematical expression"
    }

    fn usage(&self) -> &str {
        "<tool:calc>2 + 2 * 3</tool>"
    }

    fn execute(&self, args: &str) -> ToolResult {
        // Simple eval using Python (since we have it available)
        let expr = args.trim();

        match Command::new("python3")
            .arg("-c")
            .arg(format!("print({})", expr))
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    ToolResult::ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    ToolResult::err(String::from_utf8_lossy(&output.stderr).to_string())
                }
            }
            Err(e) => ToolResult::err(format!("Calc error: {}", e)),
        }
    }
}

/// Create a default tool registry with common tools
pub fn default_tools() -> crate::ToolRegistry {
    let mut registry = crate::ToolRegistry::new();
    registry.register(EchoTool);
    registry.register(ReadFileTool);
    registry.register(WriteFileTool);
    registry.register(ListDirTool);
    registry.register(ShellTool::new());
    registry.register(CalcTool);
    registry
}
