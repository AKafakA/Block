import sys
import logging
logger = logging.getLogger(__name__)

STOP_WORD_MAPS = {
    "Qwen": ["<|im_start|>", "<|im_end|>"]
}

# Ollama stream reading configuration
# Max consecutive empty reads before considering stream stalled
# At 0.001s (1ms) sleep per empty read: 1000 = 1s, 10000 = 10s, 30000 = 30s
MAX_EMPTY_READS_BEFORE_TIMEOUT = 1000  # Default: 1 second


def set_ulimit(target_soft_limit: int = 65535):
    if sys.platform.startswith("win"):
        logger.info("Skipping ulimit setting on Windows platform.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )
    new_soft, new_hard = resource.getrlimit(resource_type)
    set_ulimit_succeed = new_soft >= target_soft_limit
    return set_ulimit_succeed
