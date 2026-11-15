#!/usr/bin/env bash
# pack.sh <out dir> <platform> <app exe> <engine bin dir> <repo>: lays out the portable ToRc folder and zips it.
# The root holds the app, the README and the licences; chats, data, docs, engine, mcps, models, primers, tools and
# workspace are folders beside it. A Windows on ARM build made with llvm-mingw needs libunwind.dll and
# WebView2Loader.dll beside the app (libunwind beside the engine too): they are copied, with their licences, when
# they sit beside the app or in the engine bin dir. Linux binaries are stripped of debug info on the way in.
set -euo pipefail
out=$1 plat=$2 app=$3 bin=$4 repo=$5 PY=${PY:-python3}
ext=; [[ $plat == windows-* ]] && ext=.exe
name=ToRc-0.1.0-$plat
root=$out/$name
mkdir -p "$root"/{chats,data,docs,engine,mcps,models,primers,tools,workspace}
cp "$app" "$root/ToRc$ext"
for e in torc-server torc-run torc-tune torc hwscan; do cp "$bin/$e$ext" "$root/engine/"; done
[[ -z $ext ]] && strip --strip-debug "$root/ToRc" "$root"/engine/*
dlls=
for dll in libunwind.dll WebView2Loader.dll; do
  for from in "$(dirname "$app")" "$bin"; do
    if [[ -f $from/$dll ]]; then cp "$from/$dll" "$root/"; dlls=1; break; fi
  done
  # an exe that imports the DLL names it; an MSVC build links both in and needs neither
  if [[ ! -f $root/$dll ]] && grep -qaF "$dll" "$app"; then
    echo "$dll is in neither $(dirname "$app") nor $bin, and the app needs it beside it" >&2
    exit 1
  fi
done
[[ -f $root/libunwind.dll ]] && cp "$root/libunwind.dll" "$root/engine/"
cp "$repo"/{LICENSE-MIT,LICENSE-APACHE,NOTICE,THIRD-PARTY-NOTICES.md,THIRD-PARTY-LICENSES.md} "$root/"
cp "$repo"/desktop/package/{README.md,THIRD-PARTY-LICENSES-APP.md} "$root/"
[[ -n $dlls ]] && cp "$repo/desktop/package/THIRD-PARTY-DLLS.md" "$root/"
cp "$repo"/{GUIDE.md,AGENTS.md,SWITCHES.md,AUTOTUNE.md,TWO-BOX.md,EXPERT-CACHE.md,FORMAT.md,RESULTS.md} "$root/docs/"
if [[ -n $ext ]]; then eol=$'\r\n' torc='engine\torc.exe'; else eol=$'\n' torc=engine/torc; fi
printf "%s$eol" 'ToRc runs .torc models. Open a GGUF in the app and it converts it, or run:' \
  "  $torc convert-gguf <in.gguf> <out.torc>" > "$root/models/PUT YOUR MODELS HERE.txt"
# Python's zipfile keeps the Unix execute bits, which Compress-Archive drops.
(cd "$out" && rm -f "$name.zip" && $PY -m zipfile -c "$name.zip" "$name")
ls -la "$out/$name.zip"
