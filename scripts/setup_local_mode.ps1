param(
    [string]$Model = "llama3.1:8b",
    [string]$EmbeddingModel = "nomic-embed-text",
    [string]$OllamaBaseUrl = "http://127.0.0.1:11434",
    [string]$OllamaExePath = "",
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

function Get-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -Path $Path)) {
        return $null
    }

    $pattern = "^\s*$([Regex]::Escape($Name))\s*=\s*(.*)\s*$"
    foreach ($line in Get-Content -Path $Path) {
        if ($line -match "^\s*#") {
            continue
        }

        if ($line -match $pattern) {
            return $Matches[1].Trim()
        }
    }

    return $null
}

function Resolve-OllamaExe {
    param(
        [string]$PreferredPath,
        [string]$DotEnvPath
    )

    if ($PreferredPath -and (Test-Path -Path $PreferredPath)) {
        return $PreferredPath
    }

    if ($env:MCP_OLLAMA_EXE -and (Test-Path -Path $env:MCP_OLLAMA_EXE)) {
        return $env:MCP_OLLAMA_EXE
    }

    $envFileOllamaExe = Get-DotEnvValue -Path $DotEnvPath -Name "MCP_OLLAMA_EXE"
    if ($envFileOllamaExe -and (Test-Path -Path $envFileOllamaExe)) {
        return $envFileOllamaExe
    }

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaCmd -and (Test-Path -Path $ollamaCmd.Source)) {
        return $ollamaCmd.Source
    }

    return $null
}

function Test-OllamaApi {
    param([string]$BaseUrl)

    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/api/tags" -Method Get -TimeoutSec 5 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location -Path $repoRoot

$envFile = Join-Path $repoRoot ".env"
$envExampleFile = Join-Path $repoRoot ".env.example"

$ollamaExe = Resolve-OllamaExe -PreferredPath $OllamaExePath -DotEnvPath $envFile
if (-not $ollamaExe) {
    $ollamaExe = Resolve-OllamaExe -PreferredPath $OllamaExePath -DotEnvPath $envExampleFile
}

if (-not $ollamaExe) {
    throw "Ollama CLI not found. Set MCP_OLLAMA_EXE in .env or add Ollama to PATH."
}

Write-Host "Using Ollama executable: $ollamaExe"

if (-not (Test-OllamaApi -BaseUrl $OllamaBaseUrl)) {
    Write-Host "Ollama API is not reachable at $OllamaBaseUrl. Starting 'ollama serve'..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden | Out-Null
}

Write-Host "Pulling local LLM model: $Model"
& $ollamaExe pull $Model

Write-Host "Pulling local embedding model: $EmbeddingModel"
& $ollamaExe pull $EmbeddingModel

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
Set-Or-AddEnvLine -Path $envFile -Name "MCP_OLLAMA_EXE" -Value $ollamaExe
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
