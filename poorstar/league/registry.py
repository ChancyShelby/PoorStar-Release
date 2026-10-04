"""League: registry."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest, is_sha

ROLES = ("main", "historical", "main_exploiter", "league_exploiter", "intent_specialist", "anchor")


class SnapshotRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, weights, *, identity, policy_version, role, intent_profile=None):
        if role not in ROLES or type(policy_version) is not int or policy_version < 0:
            raise ValueError("declared snapshot role/version required")
        if intent_profile is not None:
            if (
                role != "intent_specialist"
                or not intent_profile
                or any(
                    not isinstance(v, (int, float)) or not 0 <= v <= 1
                    for v in intent_profile.values()
                )
            ):
                raise ValueError("explicit bounded Intent specialization required")
        elif role == "intent_specialist":
            raise ValueError("Intent specialist requires a declared profile")
        content = file_hash(weights)
        record = dict(
            schema="poorstar_v3_league_snapshot_v1",
            weights_sha256=content,
            identity_sha256=identity.sha256,
            source_sha256=identity.source_sha256,
            training_identity=asdict(identity),
            model_config_sha256=identity.model_config_sha256,
            scope_sha256=identity.scope_sha256,
            profile_sha256=identity.profile_sha256,
            policy_version=policy_version,
            arm=identity.arm,
            seed=identity.seed,
            matchup=identity.matchup,
            role=role,
            intent_profile=intent_profile,
        )
        snapshot_id = digest(record)
        record["snapshot_id"] = snapshot_id
        destination = self.root / snapshot_id
        if destination.exists():
            if self.read(snapshot_id) != record:
                raise ValueError("immutable snapshot collision")
            return record
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=self.root))
        try:
            with (
                Path(weights).open("rb") as source,
                (temporary / "weights.pt").open("xb") as target,
            ):
                import shutil

                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            if file_hash(temporary / "weights.pt") != content:
                raise ValueError("snapshot copy hash mismatch")
            with (temporary / "snapshot.json").open("x") as f:
                json.dump(record, f, sort_keys=True, indent=2, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.rename(temporary, destination)
        except BaseException:
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return record

    def read(self, snapshot_id):
        if not is_sha(snapshot_id):
            raise ValueError("content-addressed snapshot ID required")
        record = json.loads((self.root / snapshot_id / "snapshot.json").read_text())
        check = dict(record)
        value = check.pop("snapshot_id")
        if (
            digest(check) != value
            or value != snapshot_id
            or file_hash(self.weights_path(snapshot_id)) != record["weights_sha256"]
        ):
            raise ValueError("immutable snapshot drift")
        return record

    def weights_path(self, snapshot_id):
        if not is_sha(snapshot_id):
            raise ValueError("invalid snapshot identity")
        return self.root / snapshot_id / "weights.pt"

    def list(self):
        return tuple(
            self.read(p.name) for p in sorted(self.root.iterdir()) if p.is_dir() and is_sha(p.name)
        )
