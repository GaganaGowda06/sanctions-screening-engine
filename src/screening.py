"""The screening interface: one function, screen(name, birth_date=None, country=None).

    from screening import screen
    for match in screen("Vladimir Putin", "1952-10-07", "ru"):
        print(match.score, match.explanation)

screen() returns the entities whose final score reaches the alert threshold for the script of the
query, best first. Each match carries an explanation:
  name_score        name similarity before the skeleton and before birth date and country
  skeleton_used     True if the vowel-stripped skeleton raised the name score
  birth_date        agree, conflict, close (a year apart, no effect) or missing
  country           agree, conflict or missing
  matched_name      the entity's own spelling that the query matched best

The alert threshold is one global value from src/screening_config.json. Per-script thresholds (Latin,
Cyrillic, Arabic, other) are kept in the same file and switched on with per_script, but are off.
The list screened against is the named target entities of the downloaded OpenSanctions file, built
once on first use.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from entities import load_named_entities
from profile_data import DEFAULT_INPUT, name_scripts
from scoring import (
    DEFAULT_BLEND,
    DEFAULT_SKELETON_FLOOR,
    DEFAULT_SKELETON_LIFT,
    DEFAULT_THRESHOLD,
    Screener,
    best_matching_name,
    combine,
    country_comparison,
    date_comparison,
    name_with_skeleton,
    scoring_form,
)

SCRIPTS = ("Latin", "Cyrillic", "Arabic", "other")
CONFIG_PATH = Path(__file__).with_name("screening_config.json")
MAX_NAMES_SHOWN = 5


def query_script(name: str) -> str:
    """Script class that picks the alert threshold. Arabic wins over Cyrillic, which wins over the rest."""
    scripts = name_scripts(name)
    if "Arabic" in scripts:
        return "Arabic"
    if "Cyrillic" in scripts:
        return "Cyrillic"
    if scripts - {"Latin"}:
        return "other"
    return "Latin"


@dataclass(frozen=True)
class ScreeningConfig:
    """Scoring parameters and alert thresholds.

    By default one global threshold applies to every query. The per-script thresholds are kept in the
    configuration and used only when per_script is True: they gave no useful gain (see README).
    """

    skeleton_floor: float = DEFAULT_SKELETON_FLOOR
    skeleton_lift: float = DEFAULT_SKELETON_LIFT
    blend: float = DEFAULT_BLEND
    thresholds: dict[str, float] = field(default_factory=lambda: dict.fromkeys(SCRIPTS, DEFAULT_THRESHOLD))
    global_threshold: float = DEFAULT_THRESHOLD
    per_script: bool = False

    def threshold_for(self, script: str) -> float:
        return self.thresholds[script] if self.per_script else self.global_threshold

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ScreeningConfig":
        """The saved configuration, or the defaults if the file does not exist."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            data["skeleton_floor"],
            data["skeleton_lift"],
            data["blend"],
            {script: data["thresholds"][script] for script in SCRIPTS},
            data.get("global_threshold", DEFAULT_THRESHOLD),
            data.get("per_script", False),
        )


@dataclass(frozen=True)
class Explanation:
    script: str
    threshold: float
    matched_name: str
    name_score: float
    skeleton_score: float
    skeleton_used: bool
    birth_date: str
    country: str
    birth_date_adjustment: float
    country_adjustment: float


@dataclass(frozen=True)
class ScreenMatch:
    entity_id: str
    score: float
    names: tuple[str, ...]
    explanation: Explanation

    def as_dict(self) -> dict:
        return asdict(self)


class ScreeningEngine:
    """A screener plus the tuned configuration. screen() on the engine is what the module function calls."""

    def __init__(self, screener: Screener, config: ScreeningConfig | None = None) -> None:
        self.screener = screener
        self.config = config or ScreeningConfig.load()

    @classmethod
    def from_entities(
        cls, entities: list[dict], config: ScreeningConfig | None = None, max_df: int = 1000
    ) -> "ScreeningEngine":
        config = config or ScreeningConfig.load()
        return cls(Screener.build(entities, max_df=max_df, blend=config.blend), config)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_INPUT, config: ScreeningConfig | None = None) -> "ScreeningEngine":
        """Build from the named target entities in an OpenSanctions entities file."""
        return cls.from_entities([e for e in load_named_entities(path) if e["target"]], config)

    def screen(
        self, name: str, birth_date: str | None = None, country: str | None = None, limit: int | None = None
    ) -> list[ScreenMatch]:
        config, screener = self.config, self.screener
        script = query_script(name)
        threshold = config.threshold_for(script)
        query = scoring_form(name)
        matches = []
        for c in screener.components(name, birth_date, country, keep=10_000, blends=(config.blend,)):
            score = combine(c, config.skeleton_floor, config.skeleton_lift)
            if score < threshold:
                continue
            record = screener.records[c.entity_id]
            _, used = name_with_skeleton(c, config.skeleton_floor, config.skeleton_lift)
            date_status, date_adj = date_comparison(birth_date, record.birth_dates, screener.adjustments)
            country_status, country_adj = country_comparison(country, record.countries, screener.adjustments)
            matches.append(
                ScreenMatch(
                    entity_id=c.entity_id,
                    score=score,
                    names=record.raw_names[:MAX_NAMES_SHOWN],
                    explanation=Explanation(
                        script=script,
                        threshold=threshold,
                        matched_name=best_matching_name(query, record, screener.idf, config.blend),
                        name_score=c.names[0],
                        skeleton_score=c.skeleton,
                        skeleton_used=used,
                        birth_date=date_status,
                        country=country_status,
                        birth_date_adjustment=date_adj,
                        country_adjustment=country_adj,
                    ),
                )
            )
        matches.sort(key=lambda m: (-m.score, m.entity_id))
        return matches[:limit]


_engine: ScreeningEngine | None = None


def get_engine() -> ScreeningEngine:
    """The shared engine over the downloaded list, built on first use (a few seconds)."""
    global _engine
    if _engine is None:
        _engine = ScreeningEngine.from_file()
    return _engine


def screen(name: str, birth_date: str | None = None, country: str | None = None) -> list[ScreenMatch]:
    """Screen a name, with optional birth date (YYYY, YYYY-MM or YYYY-MM-DD) and ISO country code."""
    return get_engine().screen(name, birth_date, country)
