//! Offline tests for the Cursor Agent CLI provider.
//!
//! The fake executable is a shell script, so these tests are Unix-only.
//! They assert the argv contract: ask mode, no `--yolo` / `--force`.

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use ai_memory_llm::{
    CURSOR_DEFAULT_MODEL, ChatRequest, CursorAgentProvider, LlmProvider, ProviderAuth,
};

fn sh_single(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn fake_agent(dir: &Path, sink: &Path, stdout: &str) -> PathBuf {
    let path = dir.join("agent");
    let script = format!(
        "#!/bin/sh\nmode=$(stat -c %a . 2>/dev/null || stat -f %Lp .)\nprintf '%s\\n' \"$@\" > {}\nprintf '%s\\n' \"$mode\" >> {}\ncase \" $* \" in\n  *' --yolo '*|*' --force '*) echo refusing >&2; exit 2 ;;\nesac\nprintf '%s\\n' {}\n",
        sh_single(&sink.display().to_string()),
        sh_single(&sink.display().to_string()),
        sh_single(stdout),
    );
    std::fs::write(&path, script).unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ai-memory-cursor-test-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn provider(executable: PathBuf, model: &str) -> CursorAgentProvider {
    CursorAgentProvider::new(
        ProviderAuth::cursor(executable)
            .require_cursor_auth()
            .expect("cursor executable resolves"),
        model,
    )
}

#[tokio::test]
async fn ask_mode_returns_text_without_yolo() {
    let dir = TempDir::new();
    let sink = dir.0.join("argv");
    let executable = fake_agent(&dir.0, &sink, "OK");
    let response = provider(executable, CURSOR_DEFAULT_MODEL)
        .complete(ChatRequest::user_prompt("Reply with OK"))
        .await
        .unwrap();
    assert_eq!(response.text, "OK");
    let argv = std::fs::read_to_string(&sink).unwrap();
    assert!(argv.contains("--mode\nask\n"), "{argv}");
    assert!(argv.contains("--print\n"), "{argv}");
    assert!(!argv.contains("--yolo"), "{argv}");
    assert!(!argv.contains("--force"), "{argv}");
    assert!(argv.contains(CURSOR_DEFAULT_MODEL), "{argv}");
    assert!(argv.lines().any(|line| line == "700"), "{argv}");
}

#[tokio::test]
async fn default_model_omits_the_model_flag() {
    let dir = TempDir::new();
    let sink = dir.0.join("argv");
    let executable = fake_agent(&dir.0, &sink, "{\"answer\":\"ok\"}");
    let value = provider(executable, "default")
        .complete_structured_raw(
            ChatRequest::user_prompt("status"),
            serde_json::json!({"type": "object"}),
        )
        .await
        .unwrap();
    assert_eq!(value["answer"], "ok");
    let argv = std::fs::read_to_string(&sink).unwrap();
    assert!(!argv.contains("--model"), "{argv}");
}

#[tokio::test]
async fn fenced_json_is_parsed() {
    let dir = TempDir::new();
    let sink = dir.0.join("argv");
    let executable = fake_agent(&dir.0, &sink, "```json\n{\"answer\":1}\n```");
    let value = provider(executable, "default")
        .complete_structured_raw(
            ChatRequest::user_prompt("status"),
            serde_json::json!({"type": "object"}),
        )
        .await
        .unwrap();
    assert_eq!(value["answer"], 1);
}
