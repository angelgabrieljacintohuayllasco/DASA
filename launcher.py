"""
╔══════════════════════════════════════════════════════════════╗
║               DASA + SHARD  —  Launcher unificado            ║
╚══════════════════════════════════════════════════════════════╝

Uso:
    python launcher.py          →  menú interactivo
    python launcher.py api      →  arranca sólo la API REST
    python launcher.py query "¿Qué es Python?"  →  query directo
    python launcher.py tests    →  corre todos los tests
"""

# ── Auto-instalación ──────────────────────────────────────────────────────────
import sys
import os
import subprocess


ROOT = os.path.dirname(os.path.abspath(__file__))   # = DASA-main/
DASA_DIR = ROOT                                       # launcher está dentro de DASA-main
SHARD_DIR = os.path.join(ROOT, "..", "SHARD-main")   # hermano del repo


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *args])


def autoinstall():
    """Instala dependencias faltantes de ambos repos sin interrumpir al usuario."""
    # Agregar repos a sys.path inmediatamente (funciona sin reiniciar el proceso)
    for path in [DASA_DIR, SHARD_DIR]:
        if path not in sys.path:
            sys.path.insert(0, path)

    required = {
        "numpy": "numpy>=1.24.0",
        "sentence_transformers": "sentence-transformers>=2.6.0",
        "pytest": "pytest>=7.4.0",
    }
    api_required = {
        "fastapi": "fastapi>=0.110.0",
        "uvicorn": "uvicorn[standard]>=0.29.0",
    }

    missing_core = []
    missing_api = []

    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing_core.append(pkg)

    for mod, pkg in api_required.items():
        try:
            __import__(mod)
        except ImportError:
            missing_api.append(pkg)

    if missing_core:
        print(f"[setup] Instalando dependencias core: {missing_core}")
        _pip(*missing_core)

    if missing_api:
        print(f"[setup] Instalando dependencias API: {missing_api}")
        _pip(*missing_api)

    # Instalar repos en modo editable para que funcionen desde cualquier directorio
    for name, path in [("shard", SHARD_DIR), ("dasa", DASA_DIR)]:
        try:
            __import__(name)
        except ImportError:
            print(f"[setup] Instalando {name} desde {path}")
            _pip("-e", path)
            # Recargar sys.path tras la instalación editable
            import importlib
            import site
            importlib.invalidate_caches()


autoinstall()

# ── Imports principales (después de auto-install) ─────────────────────────────
import json
import logging
import platform
import secrets
import shutil
import signal
import textwrap
import time
from pathlib import Path

# Silenciar el monitor de tqdm para evitar el error de KeyboardInterrupt en atexit
try:
    import tqdm as _tqdm_mod
    _tqdm_mod.tqdm.monitor_interval = 0  # desactiva TMonitor antes de que arranque
except Exception:
    pass

# Logger global — usado en la API y en el pipeline
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dasa.api")


# ── API Key ───────────────────────────────────────────────────────────────────

_KEY_FILE = os.path.join(ROOT, ".dasa_api_key")


def _load_or_create_api_key() -> str:
    """Lee la API key guardada o genera una nueva y la persiste en .dasa_api_key."""
    if os.path.exists(_KEY_FILE):
        key = Path(_KEY_FILE).read_text(encoding="utf-8").strip()
        if key:
            return key
    # Generar clave segura: prefijo legible + 32 bytes hex
    key = "dasa-" + secrets.token_hex(24)
    Path(_KEY_FILE).write_text(key, encoding="utf-8")
    return key


API_KEY: str = _load_or_create_api_key()


# ── Configuración persistente ─────────────────────────────────────────────────

_CONFIG_FILE = os.path.join(ROOT, ".dasa_config.json")

_CONFIG_DEFAULTS: dict = {
    # Agent A — Retrieval
    "embedding_model":    "all-MiniLM-L6-v2",
    "top_k_fragments":    5,
    "similarity_threshold": 0.2,
    "dataset_path":       "",
    # Agent B — Synthesis
    "agent_b_mode":       "statistical",   # statistical | huggingface | ollama
    "synthesis_model":    "",              # nombre de modelo HuggingFace
    "ollama_host":        "http://localhost:11434",
    "ollama_model":       "llama3",
}


def load_config() -> dict:
    """Carga la configuración desde .dasa_config.json (crea el archivo si no existe)."""
    if os.path.exists(_CONFIG_FILE):
        try:
            saved = json.loads(Path(_CONFIG_FILE).read_text(encoding="utf-8"))
            return {**_CONFIG_DEFAULTS, **saved}
        except Exception:
            pass
    return dict(_CONFIG_DEFAULTS)


def save_config(cfg: dict) -> None:
    """Persiste la configuración en .dasa_config.json."""
    Path(_CONFIG_FILE).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# Carga inicial
CFG = load_config()


# ── Detección de terminal ─────────────────────────────────────────────────────

def detect_terminal() -> str:
    env = os.environ

    if env.get("WT_SESSION"):
        return "Windows Terminal"
    if env.get("TERM_PROGRAM") == "vscode":
        return "VS Code integrated terminal"
    if env.get("TERM_PROGRAM") == "iTerm.app":
        return "iTerm2"
    if env.get("TERM_PROGRAM") == "Apple_Terminal":
        return "macOS Terminal"
    if "PYCHARM_HOSTED" in env:
        return "PyCharm terminal"
    if env.get("SESSIONNAME") == "Console" and platform.system() == "Windows":
        if env.get("PROMPT"):
            return "CMD (Windows)"
        return "PowerShell (Windows)"
    if env.get("TERM") == "xterm-256color":
        return "xterm-256color"
    if env.get("TERM"):
        return f"Terminal ({env['TERM']})"
    shell = env.get("SHELL", "")
    if "bash" in shell:
        return "Bash"
    if "zsh" in shell:
        return "Zsh"
    if "fish" in shell:
        return "Fish"
    if platform.system() == "Windows":
        return "CMD/PowerShell (Windows)"
    return "Terminal desconocida"


# ── Colores ANSI (se desactivan si el terminal no los soporta) ────────────────

_USE_COLOR = sys.stdout.isatty() or bool(os.environ.get("WT_SESSION"))

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(t):    return _c("1",     t)
def cyan(t):    return _c("1;36",  t)
def green(t):   return _c("1;32",  t)
def yellow(t):  return _c("1;33",  t)
def red(t):     return _c("1;31",  t)
def dim(t):     return _c("2",     t)


# ── Header ────────────────────────────────────────────────────────────────────

def print_header():
    terminal = detect_terminal()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print()
    print(cyan("╔══════════════════════════════════════════════════════════════╗"))
    print(cyan("║") + bold("         DASA + SHARD  —  Sistema RAG Anti-Alucinación        ") + cyan("║"))
    print(cyan("╚══════════════════════════════════════════════════════════════╝"))
    print(f"  {dim('Python')}     : {py_ver}")
    print(f"  {dim('Terminal')}   : {terminal}")
    print(f"  {dim('Plataforma')} : {platform.system()} {platform.machine()}")
    print(f"  {dim('Raíz')}       : {ROOT}")
    print()


# ── Menú ──────────────────────────────────────────────────────────────────────

MENU_OPTIONS = [
    ("1", "Hacer una consulta  (DASA query interactivo)"),
    ("2", "Construir base de datos SHARD  desde JSON"),
    ("3", "Construir caché de embeddings"),
    ("4", "Iniciar API REST  (FastAPI / uvicorn)"),
    ("5", "Correr tests  (DASA + SHARD)"),
    ("6", "Ver estadísticas de una base SHARD"),
    ("7", "Configurar modelos  (Agente A · Agente B · Dataset)"),
    ("8", "API Key  (ver / regenerar)"),
    ("9", "Instalar / actualizar dependencias"),
    ("0", "Salir"),
]


def _config_status_line() -> str:
    """Resumen de una línea de la configuración actual para mostrarlo en el menú."""
    emb = CFG["embedding_model"] or yellow("⚠ sin modelo")
    mode = CFG["agent_b_mode"]
    if mode == "statistical":
        agb = "statistical"
    elif mode == "huggingface":
        m = CFG["synthesis_model"]
        agb = f"HuggingFace → {m}" if m else yellow("⚠ HuggingFace (sin modelo)")
    elif mode == "ollama":
        agb = f"Ollama → {CFG['ollama_model']}  @ {CFG['ollama_host']}"
    else:
        agb = mode
    return f"AgA: {cyan(emb)}  ·  AgB: {cyan(agb)}"


def print_menu():
    print(bold("  ┌─ MENÚ ──────────────────────────────────────────────────┐"))
    for key, desc in MENU_OPTIONS:
        if key == "0":
            print(bold("  │") + dim(f"  [{key}] {desc}"))
        else:
            print(bold("  │") + f"  {yellow('['+key+']')} {desc}")
    print(bold("  ├─ CONFIG ────────────────────────────────────────────────┤"))
    print(bold("  │") + f"  {_config_status_line()}")
    print(bold("  └─────────────────────────────────────────────────────────┘"))
    print()


# ── Opción 1: Query interactivo ───────────────────────────────────────────────

def cmd_query_interactive(query: str | None = None):
    from dasa.pipeline import DASAPipeline
    from dasa.config import DASAConfig

    demo_json = os.path.join(DASA_DIR, "data", "demo_dataset.json")

    # Usar dataset guardado en config, o el demo, o preguntar
    print()
    dataset_path = CFG.get("dataset_path") or demo_json
    if not os.path.exists(dataset_path):
        dataset_path = input("  Ruta al dataset JSON: ").strip()

    # synthesis_model en config solo para HuggingFace (pipeline.load lo descarga).
    # Para Ollama: se inyecta el conector manualmente despues de load().
    synthesis_model = None
    if CFG["agent_b_mode"] == "huggingface" and CFG["synthesis_model"]:
        synthesis_model = CFG["synthesis_model"]

    config = DASAConfig(
        embedding_model=CFG["embedding_model"],
        top_k_fragments=CFG["top_k_fragments"],
        similarity_threshold=CFG["similarity_threshold"],
        demo_data_path=dataset_path,
        synthesis_model=synthesis_model,
    )

    print(f"\n  {dim('Cargando pipeline...')} ", end="", flush=True)
    t0 = time.time()
    pipeline = DASAPipeline(config)
    pipeline.load(dataset_path)

    # Inyectar OllamaConnector si el modo es ollama
    if CFG["agent_b_mode"] == "ollama":
        from dasa.agent_b.llm_connector import OllamaConnector
        pipeline.agent_b._llm_callable = OllamaConnector(
            model=CFG["ollama_model"],
            host=CFG["ollama_host"],
        )
        print(green(f"listo en {time.time()-t0:.1f}s") + f"  {dim('(Ollama: '+CFG['ollama_model']+')')}")
    else:
        print(green(f"listo en {time.time()-t0:.1f}s"))
    print()

    while True:
        if query:
            q = query.strip()
            query = None  # sólo usar el arg la primera vez
        else:
            q = input(f"  {cyan('Query')} (enter vacío para volver): ").strip()
        if not q:
            break

        t0 = time.time()
        response = pipeline.run(q)
        elapsed = time.time() - t0

        print()
        print(f"  {bold('Respuesta')}:")
        for line in textwrap.wrap(response, width=70):
            print(f"    {line}")
        print(f"  {dim(f'({elapsed:.2f}s)')}")
        print()


# ── Opción 2: Construir SHARD DB ──────────────────────────────────────────────

def cmd_build_db():
    print()
    input_json = input("  Ruta al archivo JSON de entrada: ").strip()
    if not os.path.exists(input_json):
        print(red("  Error: archivo no encontrado."))
        return

    output_dir = input("  Directorio de salida para la BD: ").strip()
    shards_str = input("  Número de shards [1000]: ").strip()
    num_shards = int(shards_str) if shards_str.isdigit() else 1000
    key_field = input("  Campo clave [word]: ").strip() or "word"
    value_field = input("  Campo valor [definition]: ").strip() or "definition"

    # Usar el CLI de SHARD directamente
    cmd = [
        sys.executable, "-m", "shard", "build",
        "--input",  input_json,
        "--output", output_dir,
        "--shards", str(num_shards),
        "--key-field",   key_field,
        "--value-field", value_field,
    ]
    print()
    print(f"  {dim('Ejecutando:')} {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, cwd=SHARD_DIR)
    if result.returncode == 0:
        print(green("\n  Base de datos construida correctamente."))
    else:
        print(red("\n  Ocurrió un error durante la construcción."))


# ── Opción 3: Caché de embeddings ─────────────────────────────────────────────

def cmd_build_embeddings():
    print()
    db_dir = input("  Directorio de la BD SHARD: ").strip()
    json_src = input("  JSON fuente (enter para omitir si ya está en la BD): ").strip()

    script = os.path.join(DASA_DIR, "tools", "build_embedding_cache.py")
    cmd = [sys.executable, script, "--db", db_dir]
    if json_src:
        cmd += ["--json", json_src]

    print()
    print(f"  {dim('Ejecutando:')} {' '.join(cmd)}")
    print()
    subprocess.run(cmd)


# ── Opción 7: Configurar modelos ─────────────────────────────────────────────

def cmd_configure():
    global CFG
    print()
    print(cyan("  ╔─ CONFIGURACIÓN ────────────────────────────────────────╗"))
    print()

    # ── Agent A ────────────────────────────────────────────────────────────────
    print(bold("  AGENTE A — Retrieval (embeddings)"))
    print(f"  Modelo actual : {yellow(CFG['embedding_model'])}")
    print(f"  {dim('Ejemplos: all-MiniLM-L6-v2  /  paraphrase-multilingual-MiniLM-L12-v2')}")
    nuevo = input(f"  Nuevo modelo [enter para conservar]: ").strip()
    if nuevo:
        CFG["embedding_model"] = nuevo

    print()
    print(f"  Top-K fragmentos  : {yellow(str(CFG['top_k_fragments']))}")
    v = input(f"  Nuevo valor [enter para conservar]: ").strip()
    if v.isdigit():
        CFG["top_k_fragments"] = int(v)

    print()
    print(f"  Umbral similitud  : {yellow(str(CFG['similarity_threshold']))}  (0.0 – 1.0)")
    v = input(f"  Nuevo valor [enter para conservar]: ").strip()
    try:
        fv = float(v)
        if 0.0 <= fv <= 1.0:
            CFG["similarity_threshold"] = fv
    except ValueError:
        pass

    print()
    print(f"  Dataset path : {yellow(CFG['dataset_path'] or '(demo por defecto)')}")
    v = input(f"  Nueva ruta  [enter para conservar]: ").strip()
    if v:
        CFG["dataset_path"] = v

    # ── Agent B ────────────────────────────────────────────────────────────────
    print()
    print(bold("  AGENTE B — Synthesis (generación de respuesta)"))
    print(f"  Modo actual : {yellow(CFG['agent_b_mode'])}")
    print("  Opciones disponibles:")
    print(f"    {yellow('[1]')} statistical  — pura matemática, sin red neuronal, 0 alucinaciones")
    print(f"    {yellow('[2]')} huggingface  — modelo local de HuggingFace Hub")
    print(f"    {yellow('[3]')} ollama       — modelo en servidor Ollama (local o remoto)")

    mode_choice = input("  Elige modo [1/2/3, enter para conservar]: ").strip()
    if mode_choice == "1":
        CFG["agent_b_mode"] = "statistical"
    elif mode_choice == "2":
        CFG["agent_b_mode"] = "huggingface"
        print()
        print(f"  Modelo HuggingFace actual : {yellow(CFG['synthesis_model'] or '(ninguno)')}")
        print(f"  {dim('Ejemplo: Qwen/Qwen2.5-0.5B-Instruct')}")
        v = input("  Nombre del modelo [enter para conservar]: ").strip()
        if v:
            CFG["synthesis_model"] = v
    elif mode_choice == "3":
        CFG["agent_b_mode"] = "ollama"
        print()
        print(f"  Host Ollama actual  : {yellow(CFG['ollama_host'])}")
        v = input("  Nuevo host [enter para conservar]: ").strip()
        if v:
            CFG["ollama_host"] = v
        print(f"  Modelo Ollama actual: {yellow(CFG['ollama_model'])}")
        print(f"  {dim('Ejemplos: llama3  /  mistral  /  phi3')}")
        v = input("  Nombre del modelo [enter para conservar]: ").strip()
        if v:
            CFG["ollama_model"] = v

    save_config(CFG)
    print()
    print(green("  ✓ Configuración guardada."))
    print(f"  {dim('Archivo:')} {_CONFIG_FILE}")
    print()
    print(bold("  Resumen:"))
    for k, v in CFG.items():
        print(f"    {dim(k+':')} {v}")


# ── Opción 8: API Key ─────────────────────────────────────────────────────────

def cmd_api_key():
    global API_KEY
    print()
    print(f"  {bold('API Key actual')}")
    print(f"  {yellow(API_KEY)}")
    print(f"  {dim('Guardada en:')} {_KEY_FILE}")
    print()
    print("  ¿Qué deseas hacer?")
    print(f"  {yellow('[1]')} Copiar al portapapeles (si está disponible)")
    print(f"  {yellow('[2]')} Regenerar nueva API key")
    print(f"  {yellow('[0]')} Volver")
    print()
    choice = input("  → ").strip()

    if choice == "1":
        try:
            import subprocess as _sp
            if platform.system() == "Windows":
                _sp.run(["clip"], input=API_KEY.encode(), check=True)
            elif platform.system() == "Darwin":
                _sp.run(["pbcopy"], input=API_KEY.encode(), check=True)
            else:
                _sp.run(["xclip", "-selection", "clipboard"],
                        input=API_KEY.encode(), check=True)
            print(green("  ✓ Copiada al portapapeles."))
        except Exception:
            print(yellow("  No se pudo copiar automáticamente. Cópiala manualmente de arriba."))

    elif choice == "2":
        confirm = input("  ¿Confirmas regenerar? Los clientes deberán actualizarse. [s/N]: ").strip().lower()
        if confirm == "s":
            API_KEY = "dasa-" + secrets.token_hex(24)
            Path(_KEY_FILE).write_text(API_KEY, encoding="utf-8")
            print()
            print(green("  ✓ Nueva API key generada:"))
            print(f"  {yellow(API_KEY)}")
        else:
            print(dim("  Cancelado."))


# ── Opción 9: Instalar / actualizar dependencias ──────────────────────────────

# Paquetes requeridos por DASA + el launcher
_REQUIRED_PACKAGES = [
    # Core DASA
    ("sentence-transformers", "sentence_transformers"),
    ("numpy",                 "numpy"),
    # API REST
    ("fastapi",               "fastapi"),
    ("uvicorn[standard]",     "uvicorn"),
    ("httpx",                 "httpx"),
    # Tests
    ("pytest",                "pytest"),
]

# Opcionales (se ofrecen pero no se fuerzan)
_OPTIONAL_PACKAGES = [
    ("transformers",  "HuggingFace Transformers (modo HuggingFace de Agent B)"),
    ("torch",         "PyTorch CPU  (requerido por transformers)"),
    ("tqdm",          "tqdm  (barras de progreso)"),
]


def cmd_install_deps():
    import subprocess as _sp
    import importlib

    print()
    print(bold("  ── Instalar / actualizar dependencias ──────────────────────"))
    print()

    # ── Verificar estado actual ───────────────────────────────────────────────
    print(bold("  Paquetes requeridos:"))
    missing_required = []
    for pkg_install, pkg_import in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg_import.replace("-", "_").split("[")[0])
            status = green("✓ instalado")
        except ImportError:
            status = yellow("✗ falta")
            missing_required.append(pkg_install)
        print(f"    {status}  {pkg_install}")

    print()
    print(bold("  Paquetes opcionales:"))
    for pkg_install, desc in _OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg_install.split("[")[0])
            status = green("✓ instalado")
        except ImportError:
            status = dim("○ no instalado")
        print(f"    {status}  {dim(desc)}")

    print()

    # ── Opciones ──────────────────────────────────────────────────────────────
    print(bold("  ¿Qué quieres hacer?"))
    print(f"  {yellow('[1]')} Instalar solo los paquetes que faltan (recomendado)")
    print(f"  {yellow('[2]')} Instalar / actualizar TODOS los paquetes requeridos")
    print(f"  {yellow('[3]')} Instalar también los paquetes opcionales (HuggingFace, etc.)")
    print(f"  {dim('[0]')} Cancelar")
    print()

    choice = input("  → ").strip()
    if choice not in ("1", "2", "3"):
        print(dim("  Cancelado."))
        return

    to_install = []
    if choice == "1":
        to_install = missing_required
    elif choice == "2":
        to_install = [p for p, _ in _REQUIRED_PACKAGES]
    elif choice == "3":
        to_install = [p for p, _ in _REQUIRED_PACKAGES] + [p for p, _ in _OPTIONAL_PACKAGES]

    if not to_install:
        print(green("  ✓ Todo ya está instalado. No hay nada que hacer."))
        return

    print()
    print(bold(f"  Instalando {len(to_install)} paquete(s)..."))
    print(dim(f"  Comando: pip install --upgrade {' '.join(to_install)}"))
    print()

    result = _sp.run(
        [sys.executable, "-m", "pip", "install", "--upgrade"] + to_install,
        capture_output=False,
    )

    print()
    if result.returncode == 0:
        print(green("  ✓ Instalación completada correctamente."))
    else:
        print(yellow(f"  ⚠ pip terminó con código {result.returncode}. Revisa los mensajes de arriba."))


# ── Opción 4: API REST ────────────────────────────────────────────────────────

def cmd_start_api(host: str = "0.0.0.0", port: int = 8000):
    """Arranca el servidor FastAPI en el proceso actual (bloquea)."""
    try:
        import uvicorn
    except ImportError:
        print(red("  Falta uvicorn. Instálalo con: pip install uvicorn[standard]"))
        return

    if api_app is None:
        print(red("  No se pudo construir la API (falta fastapi/uvicorn)."))
        return

    print()
    print(green(f"  API disponible en  → http://localhost:{port}"))
    print(f"  {dim('Docs interactivas  →')} http://localhost:{port}/docs")
    print(f"  {dim('Base URL (Jan/etc) →')} http://localhost:{port}/v1")
    print()
    print(f"  {bold('API Key')} : {yellow(API_KEY)}")
    print(f"  {dim('(guardada en')} {_KEY_FILE}{dim(')')}")
    print()
    print(f"  {dim('Presiona Ctrl+C para detener')}")
    print()

    # Manejador de Ctrl+C: mensaje limpio en lugar de traceback
    def _on_sigint(sig, frame):
        print()
        print(dim("\n  [API] Señal de interrupción recibida. Cerrando servidor..."))

    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        uvicorn.run(api_app, host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, old_handler)
        print(green("  [API] Servidor detenido correctamente."))


# ── Opción 5: Tests ───────────────────────────────────────────────────────────

def cmd_run_tests():
    print()
    for name, path in [("DASA", DASA_DIR), ("SHARD", SHARD_DIR)]:
        print(cyan(f"  ── Tests {name} ──"))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=path,
        )
        status = green("PASSED") if result.returncode == 0 else red("FAILED")
        print(f"  {name}: {status}")
        print()


# ── Opción 6: Stats ───────────────────────────────────────────────────────────

def cmd_stats():
    print()
    db_dir = input("  Directorio de la BD SHARD: ").strip()
    if not os.path.exists(db_dir):
        print(red("  Error: directorio no encontrado."))
        return

    cmd = [sys.executable, "-m", "shard", "stats", "--db", db_dir]
    subprocess.run(cmd, cwd=SHARD_DIR)


# ── Definición de la API REST (FastAPI) ───────────────────────────────────────

def _build_api():
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import BaseModel
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError:
        return None

    from dasa.pipeline import DASAPipeline
    from dasa.config import DASAConfig

    app = FastAPI(
        title="DASA API",
        description=(
            "API REST para el sistema RAG anti-alucinación DASA+SHARD.\n\n"
            "Todas las respuestas están **ancladas** al corpus cargado — "
            "el sistema no puede inventar información.\n\n"
            "**Autenticación**: Bearer token en el header `Authorization`.\n"
            "Ejemplo: `Authorization: Bearer dasa-xxxx`"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rutas que no requieren autenticación
    _PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/v1/models"}

    class _ApiKeyMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path in _PUBLIC_PATHS:
                return await call_next(request)
            # Aceptar clave en Authorization: Bearer <key>  o  ?api_key=<key>
            auth = request.headers.get("Authorization", "")
            key_from_header = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
            key_from_query  = request.query_params.get("api_key", "")
            provided = key_from_header or key_from_query
            if not secrets.compare_digest(provided, API_KEY):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API key inválida o ausente."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    app.add_middleware(_ApiKeyMiddleware)

    # Estado compartido del pipeline
    _state: dict = {"pipeline": None, "dataset": None, "loaded_at": None}

    # ── Schemas ────────────────────────────────────────────────────────────────

    class LoadRequest(BaseModel):
        dataset_path: str
        embedding_model: str = "all-MiniLM-L6-v2"
        top_k_fragments: int = 5
        similarity_threshold: float = 0.2

    class QueryRequest(BaseModel):
        query: str
        dataset_path: str | None = None

    class QueryResponse(BaseModel):
        query: str
        response: str
        elapsed_ms: float
        dataset: str | None

    class StatusResponse(BaseModel):
        status: str
        dataset: str | None
        loaded_at: str | None
        terminal: str
        python: str
        platform: str

    # ── Endpoints ──────────────────────────────────────────────────────────────

    @app.get("/health", tags=["sistema"])
    def health():
        """Comprueba que la API está operativa."""
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse, tags=["sistema"])
    def status():
        """Información del sistema y estado del pipeline."""
        return StatusResponse(
            status="ready" if _state["pipeline"] else "no_dataset",
            dataset=_state["dataset"],
            loaded_at=_state["loaded_at"],
            terminal=detect_terminal(),
            python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=f"{platform.system()} {platform.machine()}",
        )

    @app.post("/load", tags=["pipeline"])
    def load_dataset(req: LoadRequest):
        """
        Carga un dataset JSON y prepara el pipeline.
        Llama a este endpoint antes de hacer queries.
        """
        if not os.path.exists(req.dataset_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {req.dataset_path}")

        config = DASAConfig(
            embedding_model=req.embedding_model,
            top_k_fragments=req.top_k_fragments,
            similarity_threshold=req.similarity_threshold,
            demo_data_path=req.dataset_path,
        )
        pipeline = DASAPipeline(config)
        pipeline.load(req.dataset_path)

        _state["pipeline"] = pipeline
        _state["dataset"] = req.dataset_path
        _state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        return {"status": "loaded", "dataset": req.dataset_path}

    @app.post("/query", response_model=QueryResponse, tags=["pipeline"])
    def query(req: QueryRequest):
        """
        Ejecuta una consulta en lenguaje natural contra el corpus cargado.

        Si el pipeline aún no está cargado y se provee `dataset_path`,
        se carga automáticamente. La respuesta está **garantizada** como
        derivada sólo del corpus — sin alucinaciones.
        """
        if _state["pipeline"] is None:
            if req.dataset_path:
                demo = req.dataset_path
                if not os.path.exists(demo):
                    raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {demo}")
                config = DASAConfig(
                    embedding_model="all-MiniLM-L6-v2",
                    top_k_fragments=5,
                    similarity_threshold=0.2,
                    demo_data_path=demo,
                )
                pipeline = DASAPipeline(config)
                pipeline.load(demo)
                _state["pipeline"] = pipeline
                _state["dataset"] = demo
                _state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                # Intentar con el dataset demo incluido en el repo
                demo = os.path.join(DASA_DIR, "data", "demo_dataset.json")
                if not os.path.exists(demo):
                    raise HTTPException(
                        status_code=400,
                        detail="Pipeline no inicializado. Llama a POST /load primero.",
                    )
                config = DASAConfig(
                    embedding_model="all-MiniLM-L6-v2",
                    top_k_fragments=5,
                    similarity_threshold=0.2,
                    demo_data_path=demo,
                )
                pipeline = DASAPipeline(config)
                pipeline.load(demo)
                _state["pipeline"] = pipeline
                _state["dataset"] = demo
                _state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        t0 = time.time()
        response = _state["pipeline"].run(req.query)
        elapsed_ms = (time.time() - t0) * 1000

        return QueryResponse(
            query=req.query,
            response=response,
            elapsed_ms=round(elapsed_ms, 1),
            dataset=_state["dataset"],
        )

    @app.post("/unload", tags=["pipeline"])
    def unload():
        """Libera el pipeline de la memoria."""
        _state["pipeline"] = None
        _state["dataset"] = None
        _state["loaded_at"] = None
        return {"status": "unloaded"}

    @app.get("/demo-queries", tags=["ejemplos"])
    def demo_queries():
        """Devuelve queries de ejemplo listos para probar con el dataset demo."""
        return {
            "queries": [
                "¿Qué es la inteligencia artificial?",
                "¿Cómo preparo huevos fritos?",
                "¿Qué es el aprendizaje automático?",
                "¿Para qué sirve Python?",
                "¿Qué es un algoritmo?",
            ],
            "hint": "Envía cualquiera de estos al endpoint POST /query como {\"query\": \"...\"}",
        }

    # ── OpenAI-compatible endpoint ─────────────────────────────────────────────

    class OAIMessage(BaseModel):
        role: str
        content: str

    class OAIRequest(BaseModel):
        model: str = "dasa"
        messages: list[OAIMessage]
        stream: bool = False
        # Campos OpenAI ignorados (aceptados para compatibilidad, no usados)
        temperature: float | None = None
        max_tokens: int | None = None
        top_p: float | None = None

    def _ensure_pipeline_loaded():
        """Carga el pipeline usando la config guardada. Fallback al demo."""
        if _state["pipeline"] is not None:
            logger.debug("[pipeline] Ya cargado: %s", _state['dataset'])
            return

        from dasa.agent_b.llm_connector import OllamaConnector

        demo_fallback = os.path.join(DASA_DIR, "data", "demo_dataset.json")
        raw_path = CFG.get("dataset_path") or ""
        logger.info("[pipeline] dataset_path en config: %r", raw_path)

        # Validar que sea un archivo JSON, no un directorio ni ruta inexistente
        if raw_path and os.path.isfile(raw_path):
            dataset = raw_path
        else:
            # El dataset_path configurado no es válido → usar demo
            if raw_path and os.path.exists(raw_path):
                # Es un directorio: ignorar y avisar en log
                logger.warning(
                    "[pipeline] dataset_path apunta a un directorio (%s). "
                    "Usando demo_dataset.json.", raw_path
                )
            dataset = demo_fallback

        if not os.path.isfile(dataset):
            logger.error("[pipeline] No se encontró dataset válido: %s", dataset)
            raise HTTPException(
                status_code=400,
                detail=(
                    "No se encontró ningún dataset. "
                    "Llama a POST /load con dataset_path, o configura la ruta "
                    "desde el menú [7] del launcher."
                ),
            )

        logger.info("[pipeline] Cargando dataset: %s", dataset)
        logger.info("[pipeline] Modo Agent B: %s", CFG['agent_b_mode'])
        logger.info("[pipeline] Modelo embeddings: %s", CFG['embedding_model'])

        # synthesis_model en config solo para HuggingFace.
        # Para Ollama: se inyecta el conector manualmente.
        synthesis_model = None
        if CFG["agent_b_mode"] == "huggingface" and CFG["synthesis_model"]:
            synthesis_model = CFG["synthesis_model"]

        config = DASAConfig(
            embedding_model=CFG["embedding_model"],
            top_k_fragments=CFG["top_k_fragments"],
            similarity_threshold=CFG["similarity_threshold"],
            demo_data_path=dataset,
            synthesis_model=synthesis_model,
        )
        pipeline = DASAPipeline(config)
        pipeline.load(dataset)
        logger.info("[pipeline] Pipeline listo. dataset=%s", dataset)

        # Inyectar OllamaConnector manualmente (evita que pipeline.load lo trate como HuggingFace)
        if CFG["agent_b_mode"] == "ollama":
            connector = OllamaConnector(
                host=CFG["ollama_host"],
                model=CFG["ollama_model"],
            )
            pipeline.agent_b._llm_callable = connector
            logger.info("[pipeline] OllamaConnector inyectado: model=%s host=%s",
                        CFG['ollama_model'], CFG['ollama_host'])

        _state["pipeline"] = pipeline
        _state["dataset"] = dataset
        _state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    @app.get("/v1/models", tags=["openai-compatible"])
    def list_models():
        """
        Lista de modelos disponibles — compatible con `GET /v1/models` de OpenAI.

        Úsalo con: `openai.models.list()` o cualquier cliente compatible.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": "dasa",
                    "object": "model",
                    "created": 1_700_000_000,
                    "owned_by": "dasa-local",
                    "description": (
                        "DASA — sistema RAG anti-alucinación. "
                        "Las respuestas están ancladas al corpus cargado."
                    ),
                }
            ],
        }

    @app.post("/v1/chat/completions", tags=["openai-compatible"])
    def chat_completions(req: OAIRequest):
        """
        Endpoint compatible con `POST /v1/chat/completions` de OpenAI.
        Soporta stream=true (SSE) y stream=false (JSON).
        """
        logger.info("[chat] Petición recibida: model=%s stream=%s messages=%d",
                    req.model, req.stream, len(req.messages))
        for i, m in enumerate(req.messages):
            logger.debug("[chat] message[%d] role=%s content=%r", i, m.role, m.content[:120])

        _ensure_pipeline_loaded()
        logger.info("[chat] Pipeline OK. dataset=%s", _state.get('dataset'))

        # Extraer el último mensaje de rol "user" y el system prompt del cliente
        user_content = ""
        system_content = ""
        for msg in reversed(req.messages):
            if msg.role == "user" and not user_content:
                user_content = msg.content.strip()
            elif msg.role == "system" and not system_content:
                system_content = msg.content.strip()

        if not user_content:
            logger.warning("[chat] No se encontró mensaje user en la petición.")
            raise HTTPException(status_code=400, detail="No se encontró mensaje con role='user'.")

        # Pasar el system prompt del cliente al modo libre del motor de síntesis,
        # para que _llm_free() respete las instrucciones del cliente (Jan, etc.)
        if system_content and hasattr(_state["pipeline"].agent_b, "_free_system_prompt"):
            _state["pipeline"].agent_b._free_system_prompt = system_content
            logger.debug("[chat] System prompt del cliente aplicado al modo libre.")

        logger.info("[chat] Query: %r", user_content)

        t0 = time.time()
        response_text = _state["pipeline"].run(user_content)
        elapsed_ms = (time.time() - t0) * 1000

        logger.info("[chat] Respuesta generada en %.0fms: %r", elapsed_ms, response_text[:200])

        if not response_text:
            response_text = "No se encontró información relevante en el corpus para esta consulta."
            logger.warning("[chat] Respuesta vacía del pipeline, usando fallback.")

        req_id = f"chatcmpl-dasa-{int(time.time()*1000)}"
        created = int(time.time())

        # ── Streaming SSE (stream=True) ────────────────────────────────────────────
        if req.stream:
            logger.info("[chat] Modo streaming SSE activado.")

            def _stream_tokens():
                # Partir el texto en palabras para simular streaming token a token
                words = response_text.split(" ")
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                # Chunk final con finish_reason=stop
                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("[chat] SSE stream completado: %d palabras.", len(words))

            return StreamingResponse(
                _stream_tokens(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Respuesta JSON estática (stream=False) ────────────────────────────────
        logger.info("[chat] Modo JSON estático.")
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_content.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_content.split()) + len(response_text.split()),
            },
            "system_fingerprint": "dasa-local",
            "x_elapsed_ms": round(elapsed_ms, 1),
        }

    return app


# Instancia de la app (importada por uvicorn como "launcher:api_app")
api_app = _build_api()


# ── Loop de menú ──────────────────────────────────────────────────────────────

def main_menu():
    print_header()
    while True:
        print_menu()
        try:
            choice = input(f"  {bold('Elige una opción')} → ").strip()
        except EOFError:
            # stdin cerrado (ej: ejecución no interactiva, pipe, IDE)
            print(dim("\n  [stdin cerrado] Inicia la API directamente con: python launcher.py api"))
            break
        except KeyboardInterrupt:
            print(dim("\n  Hasta luego."))
            break
        print()

        if choice == "1":
            cmd_query_interactive()
        elif choice == "2":
            cmd_build_db()
        elif choice == "3":
            cmd_build_embeddings()
        elif choice == "4":
            port_str = input("  Puerto [8000]: ").strip()
            port = int(port_str) if port_str.isdigit() else 8000
            cmd_start_api(port=port)
        elif choice == "5":
            cmd_run_tests()
        elif choice == "6":
            cmd_stats()
        elif choice == "7":
            cmd_configure()
        elif choice == "8":
            cmd_api_key()
        elif choice == "9":
            cmd_install_deps()
        elif choice == "0":
            print(dim("  Hasta luego."))
            break
        else:
            print(yellow("  Opción no válida. Escribe un número del menú."))
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        args = sys.argv[1:]

        if not args:
            main_menu()

        elif args[0] == "api":
            print_header()
            port = int(args[1]) if len(args) > 1 and args[1].isdigit() else 8000
            cmd_start_api(port=port)

        elif args[0] == "query":
            print_header()
            q = " ".join(args[1:]) if len(args) > 1 else None
            cmd_query_interactive(query=q)

        elif args[0] == "tests":
            print_header()
            cmd_run_tests()

        elif args[0] == "build":
            print_header()
            cmd_build_db()

        elif args[0] == "stats":
            print_header()
            cmd_stats()

    except KeyboardInterrupt:
        print(dim("\n  Hasta luego."))
        sys.exit(0)

    else:
        print(f"Uso: python launcher.py  [api|query|tests|build|stats]")
        sys.exit(1)
