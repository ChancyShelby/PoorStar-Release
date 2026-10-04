"""Actions: abilities."""


def canonical_ability(ability_id, remaps):
    """Return catalog canonical ID; missing/cyclic remaps are errors, never guessed aliases."""
    seen = set()
    current = int(ability_id)
    while True:
        if current not in remaps:
            raise ValueError("ability absent from engine catalog")
        if current in seen:
            raise ValueError("cyclic engine ability remapping")
        seen.add(current)
        target = int(remaps[current])
        if target in (0, current):
            return current
        current = target
