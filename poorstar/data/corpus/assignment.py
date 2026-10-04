"""Data: corpus: assignment."""

SCHEMA = "poorstar_c3_fixed_corpus_assignment_v1"


def select_assignment(manifest, identity, jobs):
    """Validate an already digest-verified JSON assignment before any output."""
    if set(manifest) != {"schema", "data_binding", "registered_views", "job_ids"}:
        raise ValueError("assignment fields drift")
    if manifest["schema"] != SCHEMA or manifest["data_binding"] != identity:
        raise ValueError("assignment source/plan/split/configuration drift")
    if type(manifest["registered_views"]) is not int or manifest["registered_views"] != len(jobs):
        raise ValueError("assignment full denominator drift")
    ids = manifest["job_ids"]
    if not isinstance(ids, list) or not ids or any(type(k) is not str for k in ids):
        raise ValueError("nonempty explicit assignment job identities required")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate assignment job identity")
    known = {job["job_id"]: job for job in jobs}
    if set(ids) - set(known):
        raise ValueError("unknown assignment job identity")

    chosen = set(ids)
    return [job for job in jobs if job["job_id"] in chosen]


def completion_scope(registered, requested, completed, failed, stopped):
    assigned = completed == requested and not failed and not stopped
    return dict(
        registered_views=registered,
        requested_assigned_views=requested,
        completed_views=completed,
        complete_assigned_engine_extraction=assigned,
        complete_engine_extraction=assigned and requested == registered,
    )
