"""Load configs/sources.yaml, print a summary, and validate mixture weights."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"
REQUIRED_FIELDS = ["id", "name", "type", "license", "decision", "weight"]
VALID_DECISIONS = {"use", "review", "blocked"}


def load_sources(path: Path = CONFIG_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["sources"]


def validate(sources: list[dict]) -> list[str]:
    errors = []
    for src in sources:
        missing = [field for field in REQUIRED_FIELDS if field not in src]
        if missing:
            errors.append(f"{src.get('id', '?')}: missing fields {missing}")
        if src.get("decision") not in VALID_DECISIONS:
            errors.append(f"{src.get('id')}: invalid decision '{src.get('decision')}'")
        if src.get("decision") == "blocked" and src.get("weight", 0) > 0:
            errors.append(f"{src['id']}: blocked source must have weight 0")

    total = sum(src.get("weight", 0) for src in sources)
    if abs(total - 1.0) > 1e-6:
        errors.append(f"weights sum to {total:.2f}, expected 1.00")
    return errors


def main() -> None:
    sources = load_sources()
    print(f"{'ID':<16}{'DECISION':<10}{'WEIGHT':>8}  LICENSE")
    for src in sources:
        print(f"{src['id']:<16}{src['decision']:<10}{src['weight']:>8.0%}  {src['license']}")

    errors = validate(sources)
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print("\nConfig is valid.")


if __name__ == "__main__":
    main()