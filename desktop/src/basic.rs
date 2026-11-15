//! The basic tools, built into the app and offered alongside ToolFunnel and any MCP servers: they need nothing
//! installed. "now" gives the date and time, "calc" works a sum exactly, "ctx" tells the model how full its context is,
//! and "compact" lets it ask for its own chat to be compacted (off until the user switches it on). Each can be switched
//! off on the Tools screen.

use serde_json::{Value, json};
use std::process::Command;

/// Each tool, what the model is told it does, and whether it is on before the user has chosen
pub const BASIC: [(&str, &str, bool); 5] = [
    ("now", "The current date and time.", true),
    ("calc", "Arithmetic: + - * / % ^ and brackets.", true),
    ("ctx", "How much of your context this chat has used, in tokens, and how much is left.", true),
    (
        "wake",
        concat!(
            "Sets a timer that wakes you later with a message you write to yourself, sent into this chat as a new ",
            "turn. Only while the ToRc app stays open; at most 3 waiting at once, 1 to 1440 minutes ahead."
        ),
        false,
    ),
    (
        "compact",
        "Compacts this chat once your reply is finished: the earlier turns are replaced by a summary, which frees \
         context. Use it when ctx shows the chat is nearly full.",
        false,
    ),
];

/// What the chat has used of the context, for ctx
pub struct Context {
    pub used: u64,
    pub size: u64,
}

pub fn basic_specs(s: &Value) -> Vec<Value> {
    BASIC
        .iter()
        .filter(|(name, _, on)| s["basic"][name].as_bool().unwrap_or(*on))
        .map(|(name, text, _)| {
            let parameters = if *name == "calc" {
                json!({"type": "object", "properties": {"expression": {"type": "string",
                    "description": "for example (3 + 4) * 2 ^ 8"}}, "required": ["expression"]})
            } else if *name == "wake" {
                json!({"type": "object", "properties": {
                    "minutes": {"type": "integer", "description": "how long from now, 1 to 1440"},
                    "message": {"type": "string", "description": "what you will be told when it fires"}},
                    "required": ["minutes", "message"]})
            } else {
                json!({"type": "object", "properties": {}})
            };
            json!({"type": "function", "function": {"name": name, "description": text, "parameters": parameters}})
        })
        .collect()
}

/// Runs a basic tool. "compact" only answers here: the turn sees the call and compacts the chat when it ends.
pub fn run_basic(name: &str, args: &Value, ctx: &Context) -> Result<String, String> {
    match name {
        "now" => Ok(now_text()),
        "calc" => calc(args["expression"].as_str().unwrap_or("")).map(|v| v.to_string()),
        "ctx" => Ok(context_text(ctx)),
        "compact" => Ok("The chat will be compacted as soon as this reply is finished.".into()),
        "wake" => wake(args).map(|(m, _)| format!("Set: in {} you will be woken with your message.", minutes(m))),
        _ => Err(format!("there is no tool called {name}")),
    }
}

/// A wake-up the model asked for: minutes ahead and its message, checked. The page keeps the timer and the count.
pub fn wake(args: &Value) -> Result<(u64, String), String> {
    let minutes = args["minutes"].as_u64().ok_or("minutes must be a whole number")?;
    let message = args["message"].as_str().unwrap_or("").trim().to_string();
    if !(1..=1440).contains(&minutes) {
        return Err("minutes must be from 1 to 1440".into());
    }
    if message.is_empty() {
        return Err("give the message to wake with".into());
    }
    Ok((minutes, message))
}

fn minutes(m: u64) -> String {
    if m == 1 { "1 minute".into() } else { format!("{m} minutes") }
}

fn context_text(c: &Context) -> String {
    if c.size == 0 {
        return format!("{} tokens used; the context size is not known", c.used);
    }
    let left = c.size.saturating_sub(c.used);
    format!("{} of {} tokens used ({}%), {} left", c.used, c.size, c.used * 100 / c.size, left)
}

#[cfg(windows)]
fn now_text() -> String {
    let out = crate::jobs::quiet(&mut Command::new("cmd")).args(["/C", "echo %DATE% %TIME%"]).output();
    out.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

#[cfg(not(windows))]
fn now_text() -> String {
    Command::new("date").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

/// The calc tool: numbers, brackets and + - * / % ^. Powers bind tightest and to the
/// right, and a sign binds looser than a power: -2^2 is -4.
fn calc(e: &str) -> Result<f64, String> {
    let mut t = Vec::new();
    let cs: Vec<char> = e.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            t.push(cs[st..i].iter().collect::<String>());
        } else {
            t.push(c.to_string());
            i += 1;
        }
    }
    struct P {
        t: Vec<String>,
        i: usize,
    }
    let bad = || "only numbers and + - * / % ^ ( )".to_string();
    impl P {
        fn peek(&self) -> Option<&str> {
            self.t.get(self.i).map(String::as_str)
        }
        fn next(&mut self) -> Option<String> {
            self.i += 1;
            self.t.get(self.i - 1).cloned()
        }
    }
    fn atom(p: &mut P) -> Result<f64, String> {
        match p.next().as_deref() {
            Some("(") => {
                let v = sum(p)?;
                if p.next().as_deref() != Some(")") {
                    return Err("only numbers and + - * / % ^ ( )".into());
                }
                Ok(v)
            }
            Some(n) if n.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '.') => {
                n.parse().map_err(|_| "only numbers and + - * / % ^ ( )".into())
            }
            _ => Err("only numbers and + - * / % ^ ( )".into()),
        }
    }
    fn power(p: &mut P) -> Result<f64, String> {
        let b = atom(p)?;
        if p.peek() != Some("^") {
            return Ok(b);
        }
        p.i += 1;
        Ok(b.powf(unary(p)?))
    }
    fn unary(p: &mut P) -> Result<f64, String> {
        match p.peek() {
            Some("-") => {
                p.i += 1;
                Ok(-unary(p)?)
            }
            Some("+") => {
                p.i += 1;
                unary(p)
            }
            _ => power(p),
        }
    }
    fn product(p: &mut P) -> Result<f64, String> {
        let mut v = unary(p)?;
        while let Some(op) = p.peek().filter(|o| ["*", "/", "%"].contains(o)).map(str::to_string) {
            p.i += 1;
            let r = unary(p)?;
            v = match op.as_str() {
                "*" => v * r,
                "/" => v / r,
                _ => v % r,
            };
        }
        Ok(v)
    }
    fn sum(p: &mut P) -> Result<f64, String> {
        let mut v = product(p)?;
        while let Some(op) = p.peek().filter(|o| *o == "+" || *o == "-").map(str::to_string) {
            p.i += 1;
            let r = product(p)?;
            v = if op == "+" { v + r } else { v - r };
        }
        Ok(v)
    }
    let mut p = P { t, i: 0 };
    let v = sum(&mut p)?;
    if p.i != p.t.len() {
        return Err(bad());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn compact_is_off_until_switched_on() {
        let names = |s| super::basic_specs(&s).iter().map(|t| t["function"]["name"].to_string()).collect::<Vec<_>>();
        assert_eq!(names(json!({})), ["\"now\"", "\"calc\"", "\"ctx\""]);
        assert!(super::wake(&json!({"minutes": 0, "message": "x"})).is_err());
        assert!(super::wake(&json!({"minutes": 5, "message": " "})).is_err());
        assert_eq!(super::wake(&json!({"minutes": 5, "message": "check"})), Ok((5, "check".into())));
        assert!(names(json!({"basic": {"compact": true, "now": false}})).contains(&"\"compact\"".to_string()));
    }

    #[test]
    fn ctx_says_what_is_left() {
        let c = super::Context { used: 4096, size: 16384 };
        assert_eq!(super::context_text(&c), "4096 of 16384 tokens used (25%), 12288 left");
    }

    #[test]
    fn calc_reads_as_on_paper() {
        for (e, want) in [("-2^2", -4.0), ("2^-1", 0.5), ("2^3^2", 512.0), ("(3 + 4) * 2 ^ 8", 1792.0), ("2*-3", -6.0)]
        {
            assert_eq!(super::calc(e), Ok(want), "{e}");
        }
        for e in ["", "2 +", "(1", "1)", "process.exit()", "2 3", "1.2.3", "2**3"] {
            assert!(super::calc(e).is_err(), "{e}");
        }
    }
}
