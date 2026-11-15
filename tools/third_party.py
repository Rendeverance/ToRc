"""third_party.py - writes THIRD-PARTY-LICENSES.md: every crate a release build (Windows and Linux) compiles, its
licence, and the licence texts, from cargo's own dependency tree and the crates' licence files in the cargo registry.

    python tools/third_party.py > THIRD-PARTY-LICENSES.md
    python tools/third_party.py --tree a.txt b.txt --registry DIR > THIRD-PARTY-LICENSES.md

With no arguments it runs `cargo tree` for the release targets (the GPU feature on) and reads the registry
under CARGO_HOME. --tree takes saved `cargo tree --prefix none --format "{p}|{l}|{r}"` output instead, and
--registry a folder holding one <crate>-<version> folder per crate.

Where a crate offers a choice of licences, the first of PREFER it offers is the one used here.
"""

import argparse
import os
import re
import subprocess
import sys

TARGETS = ["aarch64-pc-windows-gnullvm", "x86_64-pc-windows-gnu", "aarch64-pc-windows-msvc", "x86_64-pc-windows-msvc",
           "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
PREFER = ["MIT", "Apache-2.0", "Zlib", "ISC", "BSD-2-Clause", "Unlicense", "MIT-0", "CC0-1.0"]
NEEDS_TEXT = {"MIT", "Apache-2.0", "Zlib", "ISC", "BSD-2-Clause", "BSD-3-Clause", "Unicode-3.0", "MPL-2.0",
              "CDLA-Permissive-2.0"}
# one text serves every crate under these: the licence names no one in its body
SHARED = {"Apache-2.0", "MPL-2.0", "CDLA-Permissive-2.0"}
# licence id -> the file names crates use for it (lower case, first match wins)
FILES = {
    "MIT": ["license-mit", "license.mit", "license-mit.md", "license-mit.txt"],
    "Apache-2.0": ["license-apache", "license.apache", "license-apache-2.0", "license-apache.md", "license_a2"],
    "Zlib": ["license-zlib", "license-zlib.md"],
    "BSD-2-Clause": ["license-bsd"],
    "BSD-3-Clause": ["license.bsd-3-clause", "license-bsd-3-clause", "license-bsd"],
    "MPL-2.0": ["license-mpl", "license-mpl2", "license.mpl"],
    "Unicode-3.0": ["license-unicode"],
}
SINGLE = ["license", "license.txt", "license.md", "copying", "copyright"]
OWN = ("torc-", "hwscan ")


def tree_lines():
    out = []
    for t in TARGETS:
        r = subprocess.run(["cargo", "tree", "--workspace", "--all-features", "--target", t, "-e", "normal,build",
                            "--prefix", "none", "--format", "{p}|{l}|{r}"], capture_output=True, text=True, check=True)
        out += r.stdout.splitlines()
    return out


def registry():
    home = os.environ.get("CARGO_HOME") or os.path.join(os.path.expanduser("~"), ".cargo")
    src = os.path.join(home, "registry", "src")
    return [os.path.join(src, d) for d in os.listdir(src)]


def crates(lines):
    seen = {}
    for ln in lines:
        if "|" not in ln or ln.startswith(OWN):
            continue
        p, lic, repo = (ln.split("|") + ["", ""])[:3]
        p = p.replace(" (proc-macro)", "").replace(" (*)", "").strip()
        name, ver = p.split(" v", 1)
        # cargo marks a crate it has listed before with " (*)" at the end of the line, which is after the repository
        seen[(name, ver)] = (lic.strip().replace("/", " OR "), repo.replace(" (*)", "").strip())
    return sorted(seen.items())


def choose(expr):
    """The licences this project uses a crate under: one from each AND-ed part."""
    used = []
    for part in re.split(r"\)\s+AND\s+|\s+AND\s+\(?", expr):
        offered = [x.strip("() ") for x in part.split(" OR ")]
        used.append(next((p for p in PREFER if p in offered), offered[0]))
    return used


def text(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    try:
        return b.decode("utf-8").strip()
    except UnicodeDecodeError:              # a few older licence files are Latin-1 (the copyright sign)
        return b.decode("cp1252").strip()


def find_text(dirs, crate, lic):
    for d in dirs:
        path = os.path.join(d, crate)
        if not os.path.isdir(path):
            continue
        names = {f.lower(): os.path.join(path, f) for f in os.listdir(path)}
        for n in FILES.get(lic, []) + SINGLE:
            if n in names:
                return text(names[n])
    return None


def mit_text(dirs, crate):
    """The MIT text (ToRc's own LICENSE-MIT body) under the authors in the crate's Cargo.toml."""
    for d in dirs:
        toml = os.path.join(d, crate, "Cargo.toml")
        if os.path.isfile(toml):
            m = re.search(r"^authors\s*=\s*\[(.*?)\]", text(toml), re.M | re.S)
            if not m:
                return None
            who = ", ".join(re.findall(r'"([^"]+)"', m.group(1)))
            body = text(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LICENSE-MIT"))
            return "MIT License\n\nCopyright (c) " + who + "\n\n" + body.split("\n\n", 2)[2]
    return None


def bsd3_text(dirs, crate):
    """The BSD-3-Clause text under the authors in the crate's Cargo.toml, for a crate that ships no file."""
    for d in dirs:
        toml = os.path.join(d, crate, "Cargo.toml")
        if os.path.isfile(toml):
            m = re.search(r"^authors\s*=\s*\[(.*?)\]", text(toml), re.M | re.S)
            if not m:
                return None
            who = ", ".join(re.findall(r'"([^"]+)"', m.group(1)))
            return "Copyright (c) " + who + "\n\n" + BSD3
    return None


BSD3 = """\
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""


def notice(dirs, crate):
    for d in dirs:
        path = os.path.join(d, crate)
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().startswith("notice"):
                    return text(os.path.join(path, f))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", nargs="*")
    ap.add_argument("--registry")
    ap.add_argument("--about", help="the paragraph under the title, instead of the engine's")
    o = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    lines = []
    if o.tree:
        for t in o.tree:
            with open(t, encoding="utf-8") as f:
                lines += f.read().splitlines()
    else:
        lines = tree_lines()
    dirs = [o.registry] if o.registry else registry()
    cs = crates(lines)

    texts = {}      # text -> [crate]
    apache = {}
    missing = []
    rows = []
    for (name, ver), (expr, repo) in cs:
        crate = f"{name}-{ver}"
        used = choose(expr)
        rows.append(f"| {name} | {ver} | {expr} | {' AND '.join(used)} | {repo} |")
        for lic in used:
            if lic not in NEEDS_TEXT:
                continue
            t = find_text(dirs, crate, lic)
            if t is None and lic == "MIT":      # no file shipped: the standard text with the crate's own authors
                t = mit_text(dirs, crate)
            if t is None and lic == "BSD-3-Clause":
                t = bsd3_text(dirs, crate)
            if t is None and lic in SHARED:
                t = apache.get(lic)
            if t is None:
                missing.append(f"{name} {ver} ({lic})")
                continue
            if lic in SHARED:     # the same licence, whatever the copy's layout: one text for all of them
                t = apache.setdefault(lic, t)
            texts.setdefault(t, []).append(f"{name} {ver}")
        n = notice(dirs, crate)
        if n:
            texts.setdefault(n, []).append(f"{name} {ver} (NOTICE)")

    w = sys.stdout.write
    w("# Third-party licences\n\n")
    w((o.about or "The crates a build of ToRc compiles (Windows and Linux, ARM64 and x86-64, with the GPU feature; a "
       "few are build-time only and never reach the program), from `cargo tree`, and the licence texts they ship.") +
      " Where a crate offers a choice, the column \"used under\" is the one taken here. Generated by "
      "`tools/third_party.py`; the code and data ToRc itself borrows are in "
      "[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).\n\n")
    w(f"{len(cs)} crates.\n\n| Crate | Version | Licence | Used under | Source |\n|---|---|---|---|---|\n")
    w("\n".join(rows) + "\n\n")
    if missing:
        w("Crates that ship no file for the licence used (the standard text applies, with the authors named in the "
          "crate's Cargo.toml): " + ", ".join(missing) + ".\n\n")
    w("## Licence texts\n\n")
    for t, users in sorted(texts.items(), key=lambda x: x[1][0]):
        w("### " + ", ".join(users) + "\n\n```\n" + t.replace("```", "'''") + "\n```\n\n")


if __name__ == "__main__":
    main()
