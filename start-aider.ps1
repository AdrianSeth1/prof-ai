# start-aider.ps1 — launch aider with the local Ollama, bypassing the flaky uv shim.
# Usage:
#   .\start-aider.ps1
#   .\start-aider.ps1 ollama_chat/qwen3:30b-a3b
#   .\start-aider.ps1 ollama_chat/qwen3.6:27b --read SOMEFILE.md --file code.py
# Anything after the model name is passed straight to aider.

param(
    [string]$Model = "ollama_chat/qwen3.6:27b",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$env:OLLAMA_API_BASE = "http://127.0.0.1:11434"
Set-Location $PSScriptRoot

& "$env:APPDATA\uv\tools\aider-chat\Scripts\python.exe" -m aider --model $Model @ExtraArgs
