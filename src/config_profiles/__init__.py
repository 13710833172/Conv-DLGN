"""Config profile helpers."""

from src.config_profiles.paper_table6 import apply_paper_table6_profile


def apply_profile_if_requested(cfg: dict) -> dict:
    """Apply model profile defaults when ``model.profile_source`` is set."""
    profile = str(cfg.get("model", {}).get("profile_source", "")).strip().lower()
    if profile == "paper_table6":
        return apply_paper_table6_profile(cfg)
    return cfg


__all__ = ["apply_paper_table6_profile", "apply_profile_if_requested"]
