import os

# `settings` (omni_ingest.core.config.settings) is a module-level pydantic-settings
# singleton built at import time, before any test function runs, so these must be
# real env vars here rather than per-test monkeypatch.setenv.
os.environ.setdefault("AZURE_TRANSLATION_KEY", "dummy-translation-key")
os.environ.setdefault("AZURE_TRANSLATION_REGION", "eastus")

# AsyncSarvamAI.__init__ defaults api_subscription_key to os.getenv("SARVAM_API_KEY")
# as an eagerly-bound default argument, evaluated when sarvamai.client is first
# imported -- same import-time-binding problem as `settings` above, so this also
# can't move to a per-test monkeypatch.setenv.
os.environ.setdefault("SARVAM_API_KEY", "dummy-sarvam-key")
