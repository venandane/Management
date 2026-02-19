"""
Centralized logging with automatic __main__ resolution.
FIXES: Prevents core/logger.py from creating its own log file during initialization.
"""
import logging
import logging.handlers
from pathlib import Path
import sys
import os
import inspect

# Determine project root (parent of core directory)
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_ROOT_LOGGER_CONFIGURED = False


def _setup_root_logger():
    """Configure root logger ONCE with console handler"""
    global _ROOT_LOGGER_CONFIGURED
    if _ROOT_LOGGER_CONFIGURED:
        return
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    _ROOT_LOGGER_CONFIGURED = True


def _resolve_module_name(provided_name: str, caller_frame=None) -> str:
    """
    Resolve logger name to proper dotted path, handling __main__ case.
    
    CRITICAL FIX: Uses caller's __file__ attribute (not sys.argv) for reliability.
    Handles:
      - Direct execution: python scripts/test.py → scripts.test
      - Module execution: python -m scripts.test → scripts.test
      - Import scenarios: from scripts import test → scripts.test
    """
    if provided_name != "__main__":
        return provided_name
    
    # PRIMARY METHOD: Get __file__ from caller's global scope (most reliable)
    try:
        # If not provided, get frame of caller of get_logger (two frames back)
        if caller_frame is None:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
        
        if caller_frame and '__file__' in caller_frame.f_globals:
            script_file = caller_frame.f_globals['__file__']
            script_path = Path(script_file).resolve()
            
            try:
                rel_path = script_path.relative_to(PROJECT_ROOT)
                # Convert: scripts/strategy.py → scripts.strategy
                return str(rel_path.with_suffix('')).replace(os.sep, '.')
            except ValueError:
                # Script outside project root - use filename stem
                return script_path.stem
    except Exception:
        pass
    
    # FALLBACK: Try sys.argv[0] (less reliable but works for simple cases)
    try:
        script_path = Path(sys.argv[0]).resolve()
        if script_path.exists():
            try:
                rel_path = script_path.relative_to(PROJECT_ROOT)
                return str(rel_path.with_suffix('')).replace(os.sep, '.')
            except ValueError:
                return script_path.stem
    except Exception:
        pass
    
    # FINAL FALLBACK
    logging.getLogger("core.logger").warning(
        f"Could not resolve __main__ path. Using '__main__'. "
        f"Script may be outside project root: {PROJECT_ROOT}"
    )
    return "__main__"


def get_logger(module_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Get logger with automatic path resolution. Handles __main__ correctly.
    
    Usage in ANY file:
        from core.logger import get_logger
        logger = get_logger(__name__)
    
    Special handling:
      - When called from core/logger.py itself: uses "core.logger" (prevents recursion)
      - When called from scripts run directly: resolves to scripts.<filename>
    """
    _setup_root_logger()
    
    # CRITICAL FIX: Detect if THIS MODULE is calling get_logger (prevent self-logging during init)
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back if current_frame else None
    
    # If caller is core/logger.py itself, force name to "core.logger"
    if caller_frame and caller_frame.f_code.co_filename == __file__:
        resolved_name = "core.logger"
    else:
        resolved_name = _resolve_module_name(module_name, caller_frame)
    
    # Parse module path to create log directory structure
    parts = resolved_name.split('.')
    if len(parts) > 1:
        subdir = LOGS_DIR.joinpath(*parts[:-1])
        filename = f"{parts[-1]}.log"
    else:
        subdir = LOGS_DIR
        filename = f"{resolved_name}.log"
    
    subdir.mkdir(parents=True, exist_ok=True)
    log_path = subdir / filename
    
    # Configure logger
    logger = logging.getLogger(resolved_name)
    logger.setLevel(logging.DEBUG)
    
    # Add file handler only if not already configured
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10_000_000,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger