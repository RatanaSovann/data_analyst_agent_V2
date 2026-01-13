from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager

_FONTNAME_CACHE: dict[str, str] = {}

def _register_font(path: Path) -> str:
    key = str(path.resolve())
    if key in _FONTNAME_CACHE:
        return _FONTNAME_CACHE[key]

    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    _FONTNAME_CACHE[key] = name
    return name

def configure_fontbank(font_dir: str = "assets/fontbank") -> list[str]:
    """
    Registers all fonts in font_dir and sets matplotlib fallback chain.
    Returns the resolved family names (useful for debugging).
    """
    d = Path(font_dir)
    families: list[str] = []

    # Load every shipped font (ttf/otf/ttc)
    for p in sorted(d.glob("*.ttf")) + sorted(d.glob("*.otf")) + sorted(d.glob("*.ttc")):
        try:
            families.append(_register_font(p))
        except Exception:
            # Skip bad/unsupported font files
            pass

    # Always keep a last-resort fallback
    families.append("DejaVu Sans")

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = families
    mpl.rcParams["axes.unicode_minus"] = False

    return families