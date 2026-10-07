"""Load named entities from the OpenSanctions FollowTheMoney file, one JSON entity per line.

Standard library only.
"""

import json
from pathlib import Path

from profile_data import COUNTRY_PROPS, NAME_PROPS


def load_named_entities(path: Path) -> list[dict]:
    entities = []
    with open(path, "rb") as fh:
        for line in fh:
            try:
                ent = json.loads(line)
                props = ent.get("properties") or {}
                names = list(
                    dict.fromkeys(
                        v.strip() for p in NAME_PROPS for v in props.get(p) or () if isinstance(v, str) and v.strip()
                    )
                )
            except (ValueError, AttributeError):
                continue
            if names:
                entities.append(
                    {
                        "id": ent["id"],
                        "schema": ent.get("schema"),
                        "target": ent.get("target") is True,
                        "names": names,
                        "birth_dates": [v for v in props.get("birthDate") or () if isinstance(v, str) and v],
                        "countries": [v for p in COUNTRY_PROPS for v in props.get(p) or () if isinstance(v, str) and v],
                    }
                )
    return entities
