# main file to start the application entry point

import sys, os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.runtime_configs.config import load_config, build_runtime_cfg_from_config
from src.app.runner import run
from src.utils.logger import log

if __name__ == "__main__":
    try:
        print(f"✅ Application started. Loading configuration...")
        CONFIG, ARGS = load_config("configs/config.yaml")
        logging_cfg = CONFIG.get("logging", {})
        log.configure(logging_cfg)
        ARGS.runtime_cfg = build_runtime_cfg_from_config(CONFIG)
        try:
            run(ARGS)
        finally:
            log.close()
    except KeyboardInterrupt:
        log.info(
            "MAIN-INFO", " Program terminated by user (Ctrl+C). Exiting cleanly..."
        )
        sys.exit(0)  # exit gracefully
    except FileNotFoundError:
        log.error("MAIN-ERROR", "Configuration file 'configs/config.yaml' not found.")
        sys.exit(1)
    except Exception:
        log.error("MAIN-ERROR", "Unhandled exception occurred. Full traceback below:")
        traceback.print_exc()
        sys.exit(1)
