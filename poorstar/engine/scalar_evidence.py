"""Engine: scalar evidence."""

from collections import Counter

from poorstar.engine.observation import SCALAR_FIELDS


class ScalarEvidence:
    def __init__(self, catalog):
        self.names = {int(u.unit_id): u.name for u in catalog.units}
        if len(self.names) != len(catalog.units):
            raise ValueError("duplicate engine unit identity")
        matches = [key for key, name in self.names.items() if name == "Larva"]
        if len(matches) != 1:
            raise ValueError("unique engine Larva catalog identity required")
        self.larva_id = matches[0]
        self.fields = {name: Counter() for name in SCALAR_FIELDS}
        self.larva = Counter()
        self.selection = Counter()
        self.frames = 0

    def append(self, response):
        obs = response.observation
        if not obs.HasField("player_common") or not obs.HasField("raw_data"):
            raise ValueError("current player scalar/raw blocks required")
        common = obs.player_common
        own = [u for u in obs.raw_data.units if u.alliance == 1]
        if any(int(u.unit_type) not in self.names for u in own):
            raise ValueError("own unit type absent from engine catalog")
        for name, count in self.fields.items():
            present = common.HasField(name)
            count["present" if present else "absent"] += 1
            if present:
                count["observed_zero" if getattr(common, name) == 0 else "observed_nonzero"] += 1
        all_larvae = sum(u.unit_type == self.larva_id for u in own)
        visible_larvae = sum(u.unit_type == self.larva_id and u.display_type == 1 for u in own)
        selected = [u for u in own if u.is_selected]
        selected_larvae = sum(u.unit_type == self.larva_id for u in selected)
        present = common.HasField("larva_count")
        key = "present" if present else "absent"
        self.larva[key + "_own_larvae_positive"] += all_larvae > 0
        self.larva[key + "_own_visible_larvae_positive"] += visible_larvae > 0
        self.larva[key + "_selected_larvae_positive"] += selected_larvae > 0
        if present:
            self.larva["present_equals_own_larvae"] += common.larva_count == all_larvae
            self.larva["present_equals_own_visible_larvae"] += common.larva_count == visible_larvae
            self.larva["present_equals_selected_larvae"] += common.larva_count == selected_larvae
        types = tuple(sorted({self.names[int(u.unit_type)] for u in selected}))
        # No protocol identifiers, positions, orders or action labels enter this key.
        self.selection[(key, types)] += 1
        self.frames += 1

    def result(self):
        return dict(
            frames=self.frames,
            scalar_fields={k: dict(v) for k, v in self.fields.items()},
            larva_comparisons=dict(self.larva),
            selected_type_presence=[
                dict(presence=key, selected_types=list(types), frames=count)
                for (key, types), count in sorted(self.selection.items())
            ],
            selection_used_as_actor_feature=False,
            selection_causality_verified=False,
        )
