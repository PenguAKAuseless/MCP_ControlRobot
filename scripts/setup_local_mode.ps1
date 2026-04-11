param(
    [string]$Model = "llama3.1:8b",
    [string]$EmbeddingModel = "nomic-embed-text",
    [string]$OllamaBaseUrl = "http://127.0.0.1:11434",
    [switch]$SkipPipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-Or-AddEnvLine {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Value
    )

    $lines = @()
    if (Test-Path -Path $Path) {
        $lines = Get-Content -Path $Path
    }

    $pattern = "^$([Regex]::Escape($Name))="
    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = "$Name=$Value"
            $updated = $true
        }
    }

    if (-not $updated) {
        $lines += "$Name=$Value"
    }

    Set-Content -Path $Path -Value $lines -Encoding UTF8
}

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location -Path $repoRoot

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
$ollamaExe = $null
if ($ollamaCmd) {
    $ollamaExe = $ollamaCmd.Source
}

if (-not $ollamaExe) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -Path $candidate) {
            $ollamaExe = $candidate
            break
        }
    }
}

if (-not $ollamaExe) {
    throw "Ollama CLI not found. Install Ollama first from https://ollama.com/download"
}

Write-Host "Pulling local LLM model: $Model"
& $ollamaExe pull $Model

Write-Host "Pulling local embedding model: $EmbeddingModel"
& $ollamaExe pull $EmbeddingModel

$envFile = Join-Path $repoRoot ".env"
$envExampleFile = Join-Path $repoRoot ".env.example"

if (-not (Test-Path -Path $envFile)) {
    if (Test-Path -Path $envExampleFile) {
        Copy-Item -Path $envExampleFile -Destination $envFile
    }
    else {
        New-Item -Path $envFile -ItemType File | Out-Null
    }
}

Set-Or-AddEnvLine -Path $envFile -Name "MCP_MODE" -Value "local"
Set-Or-AddEnvLine -Path $envFile -Name "MCP_LOCAL_MODE" -Value "1"
Set-Or-AddEnvLine -Path $envFile -Name "MCP_LOCAL_BASE_URL" -Value $OllamaBaseUrl
Set-Or-AddEnvLine -Path $envFile -Name "MCP_LOCAL_LLM_MODEL" -Value $Model
Set-Or-AddEnvLine -Path $envFile -Name "MCP_LOCAL_EMBEDDING_MODEL" -Value $EmbeddingModel

if (-not $SkipPipInstall) {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -Path $venvPython) {
        Write-Host "Installing Python dependencies in .venv"
        & $venvPython -m pip install -r requirements.txt
    }
    else {
        Write-Host "Virtual environment not found; installing with current Python"
        python -m pip install -r requirements.txt
    }
}

Write-Host "Local mode setup complete."
Write-Host "Run: python mcp_pipe.py legal-answer"
