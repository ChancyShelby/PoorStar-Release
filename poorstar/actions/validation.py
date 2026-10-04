"""Actions: validation."""

from dataclasses import dataclass

from poorstar.actions.contract import TargetDomainRejection, sample, teacher_force


class SharedActionParityError(RuntimeError):
    pass


@dataclass(frozen=True)
class BundleValidation:
    status: str
    teacher: object | None
    sampled: object | None
    trace: tuple
    rejection_code: str | None = None
    rejection_message: str | None = None
    rejection_issues: tuple = ()


def assess_teacher_bundle(frame, targets):
    # Catch only the declared target exception and only in this teacher scope.
    try:
        teacher, trace = teacher_force(frame, targets)
    except TargetDomainRejection as error:
        return BundleValidation(
            "TARGET_DOMAIN_REJECTED", None, None, (), error.code, str(error), error.issues
        )
    return BundleValidation("TEACHER_ADMISSIBLE", teacher, None, trace)


def validate_shared_bundle(frame, targets):
    result = assess_teacher_bundle(frame, targets)
    if result.status == "TARGET_DOMAIN_REJECTED":
        return result
    teacher, trace = result.teacher, result.trace
    cursor = 0

    def choose(head, options, completed):
        nonlocal cursor
        if cursor >= len(trace):
            raise SharedActionParityError(
                "sampler requested a token beyond the complete teacher trace"
            )
        wanted_head, wanted_options, value = trace[cursor]
        if (head, options) != (wanted_head, wanted_options):
            raise SharedActionParityError(
                "teacher/sampler prefix domain drift at token " + str(cursor)
            )
        cursor += 1
        return value

    # Intentionally outside the teacher rejection handler: even a typed domain
    # rejection here is an implementation/parity failure, not an expert rejection.
    sampled = sample(frame, choose)
    if cursor != len(trace):
        raise SharedActionParityError("sampler did not consume the complete teacher trace")
    if teacher.commands != sampled.commands:
        raise SharedActionParityError("complete bundle sampler drift")
    teacher_wire = tuple(a.SerializeToString() for a in teacher.protocol_actions())
    sampled_wire = tuple(a.SerializeToString() for a in sampled.protocol_actions())
    if teacher_wire != sampled_wire:
        raise SharedActionParityError("teacher/sampler complete protocol wire drift")
    return BundleValidation("SHARED_PREFIX_EQUAL", teacher, sampled, trace)
