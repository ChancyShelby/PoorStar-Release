"""Examples: inspect hierarchy."""

import json
from dataclasses import asdict

from poorstar.hierarchy.state import HierarchyConfig


def main():
    profiles = {
        arm: asdict(HierarchyConfig(arm=arm))
        for arm in ("flat", "direct_task", "persistent_intent")
    }
    print(
        json.dumps(
            {
                "purpose": "CONFIGURATION_ONLY",
                "engine_started": False,
                "optimizer_created": False,
                "capability_evaluated": False,
                "profiles": profiles,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
