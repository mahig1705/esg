# Tech Debt Tracker

## Pending

- Pydantic v2 migration in config/settings.py
  - Current warning: class-based Config is deprecated in Pydantic v2 and removed in v3.
  - Action: replace inner `class Config` with `model_config = ConfigDict(...)` and import ConfigDict from pydantic.
  - Priority: Fix soon (before full v2 migration / any v3 planning).
  - Risk: hard break after deprecation window.
  - Owner: platform/core
