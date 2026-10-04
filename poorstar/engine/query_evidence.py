"""Engine: query evidence."""


def ability_answers(response, requested_tags, *, control_tags=()):
    answers = list(response.abilities)
    if len(answers) != len(requested_tags) or len(set(requested_tags)) != len(requested_tags):
        raise ValueError("query response count or request identity mismatch")
    result = {}
    for tag, answer in zip(requested_tags, answers):
        # Some absent-unit controls can explicitly return a zero identity. Never allow
        # this relaxation for a present own unit and never remap an own pointer.
        if answer.unit_tag != tag and not (tag in control_tags and answer.unit_tag == 0):
            raise ValueError("query source identity/order mismatch")
        abilities = [(int(x.ability_id), bool(x.requires_point)) for x in answer.abilities]
        if len({x[0] for x in abilities}) != len(abilities):
            raise ValueError("duplicate ability IDs in query answer")
        result[tag] = tuple(sorted(abilities))
    return result
