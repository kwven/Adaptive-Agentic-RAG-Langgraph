from __future__ import annotations

from agentic_rag.settings import get_settings
from agentic_rag.utils.logging import setup_logging, get_logger


def main() -> None:
    # 1) Load settings
    settings = get_settings()
    # 2) Setup logging as early as possible
    setup_logging()

    log = get_logger("agentic_rag")
    log.info("Starting (skeleton only)")
    log.info("ENV=%s | LOG_LEVEL=%s", settings.app.env, settings.app.log_level)
    log.info("Mistral model=%s", settings.models.chat_model)
    log.info("Data dir=%s | Vectorstore dir=%s", settings.paths.data_dir, settings.paths.vectorstore_dir)

    log.info("OK. Next: build minimal graph + baseline RAG.")


if __name__ == "__main__":
    main()
