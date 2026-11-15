//! chat.rs - ERNIE 4.5's chat format, from the chat template the model ships with: "<|begin_of_sentence|>", then a
//! system message as its text and a newline, "User: " turns ending in a newline, "Assistant: " turns ending in
//! "<|end_of_sentence|>", and "Assistant: " to open the reply. The template has no tool format, so the tools and the
//! calls are written in Qwen2.5's JSON format (chat/json_calls.rs), and the tools' answers come back as a user turn.

use crate::chat::format::{split_system, text_of, tool_answers};
use crate::chat::json_calls;
use serde_json::Value;

/// The line after the tools' answers. ERNIE 4.5 was never shown a tool's answer in training, and without this line it
/// calls the same tool again instead of answering.
const ANSWER_HINT: &str = "These are the results of your tool calls. Answer the question with them.";

pub fn prompt(msgs: &[Value], tools: &[Value]) -> String {
    let mut s = String::from("<|begin_of_sentence|>");
    let (sys, rest) = split_system(msgs);
    if let Some(sys) = json_calls::with_tools(sys, tools) {
        s += &format!("{sys}\n");
    }
    let mut i = 0;
    while i < rest.len() {
        let m = &rest[i];
        match m["role"].as_str().unwrap_or("user") {
            "user" => s += &format!("User: {}\n", text_of(m)),
            "assistant" => s += &format!("Assistant: {}{}<|end_of_sentence|>", text_of(m), json_calls::written(m)),
            "system" => s += &format!("{}\n", text_of(m)),
            "tool" => {
                s += &format!("User: {}\n{ANSWER_HINT}\n", tool_answers(rest, &mut i));
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    s + "Assistant: "
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::format::fixtures::{chat, plain, tools, want};

    // The prompt text, pinned byte for byte: the plain chat as the model's own template gives it.
    #[test]
    fn the_prompt_text() {
        assert_eq!(
            prompt(&plain(), &[]),
            want(
                r##""<|begin_of_sentence|>Be brief.\nUser: What is the weather in Paris?\nAssistant: It is 17C with light rain.<|end_of_sentence|>User: Thanks\nAssistant: ""##
            ),
            "plain"
        );
        assert_eq!(
            prompt(&chat(), &tools()),
            want(
                r##""<|begin_of_sentence|>Be brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nUser: What is the weather in Paris?\nAssistant: <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|end_of_sentence|>User: <tool_response>\n17C, light rain\n</tool_response>\nThese are the results of your tool calls. Answer the question with them.\nAssistant: It is 17C with light rain.<|end_of_sentence|>User: Thanks\nAssistant: ""##
            ),
            "tools"
        );
    }
}
