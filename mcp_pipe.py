"""
Simple MCP stdio <-> WebSocket pipe with optional unified config.
Version: 0.2.0

Usage (env):
    export MCP_ENDPOINT=wss://api.xiaozhi.me/mcp/?token=eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjcwNjI2OSwiYWdlbnRJZCI6MTE4MTM4NSwiZW5kcG9pbnRJZCI6ImFnZW50XzExODEzODUiLCJwdXJwb3NlIjoibWNwLWVuZHBvaW50IiwiaWF0IjoxNzc1NjE4MTc2LCJleHAiOjE4MDcxNzU3NzZ9.5-Xlby6DQOMoiZKbRM6JBeED-p99PhoPk8qpu06XCyN5158Ff2Rz1bHu_hR9aiHHPDIXM8Kuo9o0DWdEd0Yl7w
    # Windows (PowerShell): $env:MCP_ENDPOINT = "<ws_endpoint>"

Start server process(es) from config:
Run all configured servers (default)
    python mcp_pipe.py

Run a single local server script (back-compat)
    python mcp_pipe.py path/to/server.py

Config discovery order:
    $MCP_CONFIG, then ./mcp_config.json

Env overrides:
    (none for proxy; uses current Python: python -m mcp_proxy)
"""

import asyncio
import websockets
import subprocess
import logging
import os
import signal
import sys
import json
import shutil
from urllib.parse import urlparse
from dotenv import load_dotenv

# Auto-load environment variables from a .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MCP_PIPE')

# Reconnection settings
INITIAL_BACKOFF = 1  # Initial wait time in seconds
MAX_BACKOFF = 600  # Maximum wait time in seconds
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate_endpoint_url(endpoint_url):
    """Validate MCP endpoint URL and return (is_valid, message)."""
    if not endpoint_url:
        return False, "Please set the `MCP_ENDPOINT` environment variable"

    parsed = urlparse(endpoint_url.strip())
    if parsed.scheme not in {"ws", "wss"}:
        return False, "`MCP_ENDPOINT` must start with ws:// or wss://"

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "`MCP_ENDPOINT` is missing a valid hostname"

    placeholder_hosts = {
        "your-mcp-endpoint",
        "example.com",
        "api.example.com",
    }
    if hostname in placeholder_hosts:
        return (
            False,
            f"`MCP_ENDPOINT` points to a placeholder host ('{hostname}'). Configure a real WebSocket endpoint URL.",
        )

    return True, ""

async def connect_with_retry(uri, target):
    """Connect to WebSocket server with retry mechanism for a given server target."""
    reconnect_attempt = 0
    backoff = INITIAL_BACKOFF
    while True:  # Infinite reconnection
        try:
            if reconnect_attempt > 0:
                logger.info(f"[{target}] Waiting {backoff}s before reconnection attempt {reconnect_attempt}...")
                await asyncio.sleep(backoff)

            # Attempt to connect
            await connect_to_server(uri, target)
            # Reset retry counters after a successful session.
            reconnect_attempt = 0
            backoff = INITIAL_BACKOFF

        except Exception as e:
            reconnect_attempt += 1
            logger.warning(f"[{target}] Connection closed (attempt {reconnect_attempt}): {e}")
            # Calculate wait time for next reconnection (exponential backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

async def connect_to_server(uri, target):
    """Connect to WebSocket server and pipe stdio for the given server target."""
    try:
        logger.info(f"[{target}] Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            logger.info(f"[{target}] Successfully connected to WebSocket server")

            # Start server process (built from CLI arg or config)
            cmd, env = build_server_command(target)
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                text=True,
                env=env
            )
            logger.info(f"[{target}] Started server process: {' '.join(cmd)}")

            # Create two tasks: read from WebSocket and write to process, read from process and write to WebSocket
            await asyncio.gather(
                pipe_websocket_to_process(websocket, process, target),
                pipe_process_to_websocket(process, websocket, target),
                pipe_process_stderr_to_terminal(process, target)
            )
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"[{target}] WebSocket connection closed: {e}")
        raise  # Re-throw exception to trigger reconnection
    except Exception as e:
        logger.error(f"[{target}] Connection error: {e}")
        raise  # Re-throw exception
    finally:
        # Ensure the child process is properly terminated
        if 'process' in locals():
            logger.info(f"[{target}] Terminating server process")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            logger.info(f"[{target}] Server process terminated")

async def pipe_websocket_to_process(websocket, process, target):
    """Read data from WebSocket and write to process stdin"""
    try:
        while True:
            # Read message from WebSocket
            message = await websocket.recv()
            logger.debug(f"[{target}] << {message[:120]}...")

            # Write to process stdin (in text mode)
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            process.stdin.write(message + '\n')
            process.stdin.flush()
    except Exception as e:
        logger.error(f"[{target}] Error in WebSocket to process pipe: {e}")
        raise  # Re-throw exception to trigger reconnection
    finally:
        # Close process stdin
        if not process.stdin.closed:
            process.stdin.close()

async def pipe_process_to_websocket(process, websocket, target):
    """Read data from process stdout and send to WebSocket"""
    try:
        while True:
            # Read data from process stdout
            data = await asyncio.to_thread(process.stdout.readline)

            if not data:  # If no data, the process may have ended
                logger.info(f"[{target}] Process has ended output")
                break

            # Send data to WebSocket
            logger.debug(f"[{target}] >> {data[:120]}...")
            # In text mode, data is already a string, no need to decode
            await websocket.send(data)
    except Exception as e:
        logger.error(f"[{target}] Error in process to WebSocket pipe: {e}")
        raise  # Re-throw exception to trigger reconnection

async def pipe_process_stderr_to_terminal(process, target):
    """Read data from process stderr and print to terminal"""
    try:
        while True:
            # Read data from process stderr
            data = await asyncio.to_thread(process.stderr.readline)

            if not data:  # If no data, the process may have ended
                logger.info(f"[{target}] Process has ended stderr output")
                break

            # Print stderr data to terminal (in text mode, data is already a string)
            sys.stderr.write(data)
            sys.stderr.flush()
    except Exception as e:
        logger.error(f"[{target}] Error in process stderr pipe: {e}")
        raise  # Re-throw exception to trigger reconnection

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Received interrupt signal, shutting down...")
    sys.exit(0)


def get_config_path():
    configured_path = os.environ.get("MCP_CONFIG", "").strip()
    if configured_path:
        return os.path.abspath(configured_path)
    return os.path.join(SCRIPT_DIR, "mcp_config.json")


def _resolve_relative_to_config(path_value, config_path):
    if os.path.isabs(path_value):
        return path_value

    config_dir = os.path.dirname(config_path)
    from_config = os.path.normpath(os.path.join(config_dir, path_value))
    if os.path.exists(from_config):
        return from_config

    from_cwd = os.path.abspath(path_value)
    if os.path.exists(from_cwd):
        return from_cwd

    return from_config


def _resolve_python_script_args(args, config_path):
    resolved_args = [str(arg) for arg in args]
    expect_module_name = False

    for index, arg in enumerate(resolved_args):
        if expect_module_name:
            expect_module_name = False
            continue

        if arg == "-m":
            expect_module_name = True
            continue

        if arg.startswith("-"):
            continue

        if arg.endswith(".py") or "/" in arg or "\\" in arg:
            resolved_args[index] = _resolve_relative_to_config(arg, config_path)
        break

    return resolved_args

def load_config():
    """Load JSON config from $MCP_CONFIG or ./mcp_config.json. Return dict or {}."""
    path = get_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config {path}: {e}")
        return {}


def build_server_command(target=None):
    """Build [cmd,...] and env for the server process for a given target.

    Priority:
    - If target matches a server in config.mcpServers: use its definition
    - Else: treat target as a Python script path (back-compat)
    If target is None, read from sys.argv[1].
    """
    if target is None:
        assert len(sys.argv) >= 2, "missing server name or script path"
        target = sys.argv[1]
    config_path = get_config_path()
    cfg = load_config()
    servers = cfg.get("mcpServers", {}) if isinstance(cfg, dict) else {}

    if target in servers:
        entry = servers[target] or {}
        if entry.get("disabled"):
            raise RuntimeError(f"Server '{target}' is disabled in config")
        typ = (entry.get("type") or entry.get("transportType") or "stdio").lower()

        # environment for child process
        child_env = os.environ.copy()
        for k, v in (entry.get("env") or {}).items():
            child_env[str(k)] = str(v)

        if typ == "stdio":
            command = entry.get("command")
            args = [str(arg) for arg in (entry.get("args") or [])]
            if not command:
                raise RuntimeError(f"Server '{target}' is missing 'command'")

            resolved_command = str(command)
            if resolved_command.lower() in {"python", "python3", "py"}:
                resolved_command = sys.executable
                args = _resolve_python_script_args(args, config_path)
            elif not os.path.isabs(resolved_command):
                found = shutil.which(resolved_command)
                if found:
                    resolved_command = found
                else:
                    resolved_command = _resolve_relative_to_config(resolved_command, config_path)

            return [resolved_command, *args], child_env

        if typ in ("sse", "http", "streamablehttp"):
            url = entry.get("url")
            if not url:
                raise RuntimeError(f"Server '{target}' (type {typ}) is missing 'url'")
            # Unified approach: always use current Python to run mcp-proxy module
            cmd = [sys.executable, "-m", "mcp_proxy"]
            if typ in ("http", "streamablehttp"):
                cmd += ["--transport", "streamablehttp"]
            # optional headers: {"Authorization": "Bearer xxx"}
            headers = entry.get("headers") or {}
            for hk, hv in headers.items():
                cmd += ["-H", hk, str(hv)]
            cmd.append(url)
            return cmd, child_env

        raise RuntimeError(f"Unsupported server type: {typ}")

    # Fallback to script path (back-compat)
    script_path = target
    if not os.path.isabs(script_path):
        script_path = _resolve_relative_to_config(script_path, config_path)
    if not os.path.exists(script_path):
        raise RuntimeError(
            f"'{target}' is neither a configured server nor an existing script"
        )
    return [sys.executable, script_path], os.environ.copy()

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Get token from environment variable or command line arguments
    endpoint_url = os.environ.get('MCP_ENDPOINT', '').strip()
    endpoint_ok, endpoint_error = validate_endpoint_url(endpoint_url)
    if not endpoint_ok:
        logger.error(endpoint_error)
        sys.exit(1)

    # Determine target: default to all if no arg; single target otherwise
    target_arg = sys.argv[1] if len(sys.argv) >= 2 else None

    async def _main():
        if not target_arg:
            cfg = load_config()
            servers_cfg = (cfg.get("mcpServers") or {})
            all_servers = list(servers_cfg.keys())
            enabled = [name for name, entry in servers_cfg.items() if not (entry or {}).get("disabled")]
            skipped = [name for name in all_servers if name not in enabled]
            if skipped:
                logger.info(f"Skipping disabled servers: {', '.join(skipped)}")
            if not enabled:
                raise RuntimeError("No enabled mcpServers found in config")
            logger.info(f"Starting servers: {', '.join(enabled)}")
            tasks = [asyncio.create_task(connect_with_retry(endpoint_url, t)) for t in enabled]
            # Run all forever; if any crashes it will auto-retry inside
            await asyncio.gather(*tasks)
        else:
            cfg = load_config()
            servers_cfg = (cfg.get("mcpServers") or {}) if isinstance(cfg, dict) else {}
            if os.path.exists(target_arg) or target_arg in servers_cfg:
                await connect_with_retry(endpoint_url, target_arg)
            else:
                logger.error("Argument must be a configured server name or local Python script path.")
                sys.exit(1)

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program execution error: {e}")
