"""Collect two-player trajectories with a shared game clock."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace

import torch

from poorstar.common.parameters import parameter_digest
from poorstar.training.contracts import RolloutSequence, RolloutStep
from poorstar.training.scorer import PolicyStepPayload


def policy_context_digest(
    observation,
    model,
    hidden,
    control_state,
    *,
    feedback=(),
    dispatch_observation=None,
    task_allocations=None,
    parameter_sha256=None,
):
    """Opaque receipts retain real lifecycle/feedback identities. A conservative
    mismatch blocks pairing; no event IDs or recurrent bytes are discarded to
    manufacture matching states. This function consumes no sampling RNG.
    """
    from dataclasses import asdict
    from hashlib import sha256

    from poorstar.evaluation.interventions import visible_input_digest
    from poorstar.training.contracts import digest, is_sha

    if parameter_sha256 is None:
        parameter_sha256 = parameter_digest(model.state_dict())
    if not is_sha(parameter_sha256):
        raise ValueError("verified immutable actor parameter hash required")
    memory = None
    if hidden is not None:
        value = hidden.detach().cpu().contiguous()
        memory = dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            bytes_sha256=sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    return digest(
        dict(
            visible_input_sha256=visible_input_digest(observation),
            observation_loop=observation["observation_loop"],
            hidden=memory,
            model_config_sha256=model.config.sha256,
            hierarchy_config_sha256=model.hierarchy_config.sha256,
            parameter_sha256=parameter_sha256,
            prior_control_state=control_state.to_dict(),
            feedback=[asdict(v) for v in feedback],
            dispatch_observation=None
            if dispatch_observation is None
            else asdict(dispatch_observation),
            task_allocations=task_allocations,
        )
    )


class SC2PairBackend:
    """Owns exactly the two PySC2 processes created by this context."""

    def __init__(self, *, map_name, version, seed, deadline=None):
        self.map_name, self.version, self.seed = map_name, version, seed
        self.env = None
        self.controllers = ()
        self.pool = None
        self.owned_process_results = []
        self.deadline = deadline

    def check_budget(self):
        if self.deadline is not None:
            remaining = self.deadline.remaining()
            for controller in self.controllers:
                controller._client._sock.settimeout(remaining)

    def __enter__(self):
        from pysc2.env import sc2_env
        from pysc2.lib import features

        startup_timeout = None
        if self.deadline is not None:
            # PySC2 uses this supported timeout flag during process connection
            # and setup, before the controller handles are available here.
            import math

            from absl import flags

            startup_timeout = flags.FLAGS["sc2_timeout"].value
            flags.FLAGS["sc2_timeout"].value = max(1, math.ceil(self.deadline.remaining()))
        try:
            interface = features.AgentInterfaceFormat(
                use_raw_units=True,
                use_raw_actions=True,
                raw_resolution=256,
                send_observation_proto=True,
            )
            self.env = sc2_env.SC2Env(
                map_name=self.map_name,
                players=[
                    sc2_env.Agent(sc2_env.Race.zerg, name="learner"),
                    sc2_env.Agent(sc2_env.Race.zerg, name="opponent"),
                ],
                agent_interface_format=interface,
                random_seed=self.seed,
                version=self.version,
                realtime=False,
                disable_fog=False,
                step_mul=1,
                visualize=False,
            )
            self.controllers = tuple(self.env._controllers)
            self.check_budget()
            self.env.reset()
            if len(self.controllers) != 2:
                raise ValueError("owned two-participant environment required")
            self.pool = ThreadPoolExecutor(max_workers=2)
            self.infos = tuple(c.game_info() for c in self.controllers)
            self.catalogs = tuple(c.data_raw() for c in self.controllers)
            if self.catalogs[0].SerializeToString(deterministic=True) != self.catalogs[
                1
            ].SerializeToString(deterministic=True):
                raise ValueError("paired engine catalog mismatch")
            return self
        except BaseException:
            if self.deadline is not None:
                self.deadline.disarm()
            self.close()
            raise
        finally:
            if startup_timeout is not None:
                flags.FLAGS["sc2_timeout"].value = startup_timeout

    def observe(self):
        self.check_budget()
        return tuple(self.pool.map(lambda c: c.observe(), self.controllers))

    def advance(self, loops):
        self.check_budget()
        if type(loops) is not int or loops < 1:
            raise ValueError("strictly positive joint game-loop advance required")
        # Materialize futures/results: exception in either player cannot leave
        # an unawaited controller step at a later scheduling boundary.
        futures = [self.pool.submit(c.step, loops) for c in self.controllers]
        errors = []
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        if self.env is not None:
            processes = [p._proc for p in self.env._sc2_procs if p._proc is not None]
            self.env.close()
            self.env = None
            self.owned_process_results = [dict(pid=p.pid, exit_code=p.poll()) for p in processes]
            if any(p["exit_code"] is None for p in self.owned_process_results):
                raise ValueError("owned SC2 process remained alive after close")
        self.controllers = ()

    def __exit__(self, *_):
        if self.deadline is not None:
            self.deadline.disarm()
        self.close()


def _natural_reward(raw, player_id):
    from s2clientprotocol import sc2api_pb2 as sc

    values = [r for r in raw.player_result if r.player_id == player_id]
    if len(values) != 1:
        raise ValueError("terminal controlled-player result unresolved")
    name = sc.Result.Name(values[0].result)
    if name not in ("Victory", "Defeat", "Tie"):
        raise ValueError("natural terminal result unresolved: " + name)
    return {"Victory": 1.0, "Defeat": -1.0, "Tie": 0.0}[name]


def collect_pair(
    backend,
    models,
    profiles,
    identity,
    *,
    policy_versions=(0, 0),
    maximum_loops,
    provenance="learner",
    episode_id=None,
    deterministic=False,
    initial_overrides=(None, None),
    maximum_resident_steps=None,
    external_programs=(None, None),
    measurement_sink=None,
    measurement_period=None,
    deterministic_players=None,
    check_budget=None,
):
    """Collect a complete/truncated game; only controlled learner's trace is trained.

    No automatic process creation or launch authorization occurs here. Callers must
    validate the corresponding phase or SELFPLAY/LEAGUE evidence before invoking
    the real backend. Mechanism tests use
    an explicitly labelled protocol backend, never a fake game-strength result.
    """
    from s2clientprotocol import query_pb2

    from poorstar.actions.coarse import coarse_sample
    from poorstar.actions.frame import ActionFrame, BoundQuery, SchedulerFrame
    from poorstar.actions.timing import PreparedBody, dispatch_live_bundle
    from poorstar.engine.observation import ExtractionContext
    from poorstar.hierarchy.session import HierarchyPolicySession
    from poorstar.hierarchy.state import ExecutionFeedback

    if maximum_loops < 1 or len(models) != 2 or len(profiles) != 2:
        raise ValueError("paired models and positive actual frame budget required")
    resident_limit = maximum_loops if maximum_resident_steps is None else maximum_resident_steps
    if type(resident_limit) is not int or resident_limit < 1:
        raise ValueError("positive resident trajectory step bound required")
    if provenance not in ("learner", "evaluator"):
        raise ValueError("explicit collector provenance required")
    if type(deterministic) is not bool:
        raise ValueError("explicit boolean actor sampling mode required")
    if deterministic_players is None:
        deterministic_players = (deterministic, deterministic)
    if (
        not isinstance(deterministic_players, tuple)
        or len(deterministic_players) != 2
        or any(type(v) is not bool for v in deterministic_players)
    ):
        raise ValueError("two explicit boolean player sampling modes required")
    if measurement_sink is not None and (
        provenance != "evaluator" or type(measurement_period) is not int or measurement_period < 1
    ):
        raise ValueError(
            "independent evaluator requires a frozen positive physical measurement clock"
        )
    episode = episode_id or "selfplay-" + str(uuid.uuid4())
    sessions = [
        HierarchyPolicySession(m, p, deterministic=d)
        for m, p, d in zip(models, profiles, deterministic_players)
    ]
    for m, session in zip(models, sessions):
        m.eval()
        session.reset(episode)
    if len(initial_overrides) != 2:
        raise ValueError("one declared initial intervention per player required")
    if len(external_programs) != 2 or any(
        p is not None and o is not None for p, o in zip(external_programs, initial_overrides)
    ):
        raise ValueError("one explicit external program or initial override per player required")
    for session, override in zip(sessions, initial_overrides):
        if override is not None:
            session.intervene(override)
    parameter_hashes = tuple(parameter_digest(m.state_dict()) for m in models)
    behavior_hash = parameter_hashes[0]
    budget = (lambda: None) if check_budget is None else check_budget
    budget()
    raws = backend.observe()
    ids = [int(r.observation.player_common.player_id) for r in raws]
    if len(set(ids)) != 2:
        raise ValueError("two distinct controlled player perspectives required")
    contexts = [ExtractionContext(pid, backend.version, "LIVE_PLAYER", False) for pid in ids]
    from poorstar.evaluation.interventions import visible_input_digest
    from poorstar.training.contracts import digest

    # These are actual initial neural-input receipts, before any player action
    # or strategic decision. Equal engine seeds alone never establish pairing.
    initial_inputs = []
    for i, raw in enumerate(raws):
        initial = SchedulerFrame.from_protocol(
            raw, backend.infos[i], profiles[i].catalog, context=contexts[i], episode=episode
        )
        initial_inputs.append(
            dict(
                player_id=ids[i],
                loop=initial.key.loop,
                visible_input_sha256=visible_input_digest(initial.policy_observation()),
                policy_context_sha256=policy_context_digest(
                    initial.policy_observation(),
                    models[i],
                    sessions[i].hidden,
                    sessions[i].control_state,
                    task_allocations={} if models[i].hierarchy_config.grounded_tasks else None,
                    parameter_sha256=parameter_hashes[i],
                ),
                pre_intervention_action_prefix_sha256=digest([]),
            )
        )
    drafts = [None, None]
    due = [int(r.observation.game_loop) for r in raws]
    phases = ["scheduler", "scheduler"]
    steps = []
    receipts = []
    observed_loops = 0
    goal_observations = []
    interventions = []
    action_prefixes = [[], []]
    next_measurement = 0 if measurement_sink is not None else maximum_loops

    def observe_goals(index, frame):
        if models[index].hierarchy_config.grounded_tasks:
            from dataclasses import asdict

            outcomes = sessions[index].observe_tasks(frame)
            parents = {t.task_id: t.parent_intent for t in sessions[index].control_state.tasks}
            goal_observations.extend(
                dict(
                    player_id=ids[index],
                    parent_revision=parents.get(outcome.task_id),
                    **asdict(outcome),
                )
                for outcome in outcomes
            )

    def finalize(
        raw,
        *,
        terminated=False,
        truncated=False,
        next_value=0.0,
        next_manager_value=0.0,
        boundary_reason="",
    ):
        draft = drafts[0]
        if draft is None:
            raise ValueError("episode ended before learner decision")
        frame, body, trace = draft["frame"], draft["body"], draft["trace"]
        end = int(raw.observation.game_loop)
        if end <= frame.key.loop:
            raise ValueError("zero-duration terminal/interruption cannot form a PPO decision")
        interrupted = trace["decision"] == "ACT" and body is None
        payload = PolicyStepPayload(
            frame,
            body,
            deepcopy(trace),
            interrupted,
            tuple(draft["results"]),
            draft["memory_before"],
        )
        reward = _natural_reward(raw, ids[0]) if terminated else 0.0
        step = RolloutStep(
            episode,
            len(steps),
            frame.key.loop,
            end,
            policy_versions[0],
            behavior_hash,
            payload.observation_sha256,
            payload.mask_sha256,
            payload,
            trace["action_log_prob"],
            trace["manager_log_prob"],
            trace["value"],
            trace["manager_value"],
            next_value,
            next_manager_value,
            reward,
            end if terminated else None,
            terminated,
            truncated,
            "engine_terminal" if terminated else boundary_reason if truncated else "",
            manager_decision=trace["manager_decision"],
            provenance=provenance,
            behavior="deterministic" if deterministic_players[0] else "sampled",
        )
        step.validate()
        steps.append(step)

    while True:
        budget()
        loops = [int(r.observation.game_loop) for r in raws]
        if len(set(loops)) != 1:
            raise ValueError("paired raw observation clocks differ")
        now = loops[0]
        observed_loops = now
        if measurement_sink is not None and (
            now == next_measurement or any(r.player_result for r in raws) or now >= maximum_loops
        ):
            frame = SchedulerFrame.from_protocol(
                raws[0], backend.infos[0], profiles[0].catalog, context=contexts[0], episode=episode
            )
            measurement_sink.observe(frame)
            next_measurement = now + measurement_period
        terminal = [bool(r.player_result) for r in raws]
        if any(terminal):
            if not all(terminal) or _natural_reward(raws[0], ids[0]) != -_natural_reward(
                raws[1], ids[1]
            ):
                raise ValueError("paired terminal result disagreement")
            finalize(raws[0], terminated=True)
            break
        resident_cut = drafts[0] is not None and len(steps) + 1 >= resident_limit
        if now >= maximum_loops or resident_cut:
            with torch.no_grad():
                frame = SchedulerFrame.from_protocol(
                    raws[0],
                    backend.infos[0],
                    profiles[0].catalog,
                    context=contexts[0],
                    episode=episode,
                )
                observe_goals(0, frame)
                v, mv = models[0].evaluate_values(
                    frame.policy_observation(),
                    sessions[0].hidden,
                    sessions[0].control_state,
                    feedback=tuple(sessions[0].pending_feedback),
                    dispatch_observation=sessions[0].dispatch_observation,
                    task_allocations=sessions[0].task_observer.current_allocations(frame)
                    if sessions[0].task_observer is not None
                    else None,
                )
            finalize(
                raws[0],
                truncated=True,
                next_value=float(v),
                next_manager_value=float(mv),
                boundary_reason="step_cap" if now >= maximum_loops else "resident_step_cap",
            )
            break
        # Scheduler/body events at this loop are handled for both players before
        # either controller advances. Scheduler may choose ACT delay=0.
        for i in range(2):
            if due[i] != now:
                continue
            session, model, profile = sessions[i], models[i], profiles[i]
            if phases[i] == "scheduler":
                frame = SchedulerFrame.from_protocol(
                    raws[i], backend.infos[i], profile.catalog, context=contexts[i], episode=episode
                )
                observe_goals(i, frame)
                if external_programs[i] is not None:
                    override = external_programs[i].override_for(now, session.control_state)
                    if override is not None:
                        from dataclasses import asdict

                        from poorstar.evaluation.interventions import visible_input_digest
                        from poorstar.training.contracts import digest

                        interventions.append(
                            dict(
                                player_id=ids[i],
                                actual_scheduler_loop=now,
                                intervention=asdict(override),
                                visible_input_sha256=visible_input_digest(
                                    frame.policy_observation()
                                ),
                                policy_context_sha256=policy_context_digest(
                                    frame.policy_observation(),
                                    model,
                                    session.hidden,
                                    session.control_state,
                                    feedback=tuple(session.pending_feedback),
                                    dispatch_observation=session.dispatch_observation,
                                    task_allocations=session._allocations(
                                        frame.policy_observation()
                                    ),
                                    parameter_sha256=parameter_hashes[i],
                                ),
                                pre_intervention_action_prefix_sha256=digest(action_prefixes[i]),
                            )
                        )
                        session.intervene(override)
                memory_before = (
                    None if session.hidden is None else session.hidden.detach().cpu().clone()
                )
                decision, delay = session.choose_schedule(
                    frame.policy_observation(), model.config.maximum_delay
                )
                if i == 0 and measurement_sink is not None:
                    c = model.hierarchy_config
                    trace = session.audit_trace
                    manager = trace["scheduler_control"]["manager"]
                    if (
                        hasattr(measurement_sink, "confidence")
                        and c.arm == "persistent_intent"
                        and c.persistence
                        and c.manager_mode == "learned"
                        and c.commitment_mode == "minimum_hold"
                        and trace["external_override"] is None
                        and manager is not None
                        and manager["operation"] == "update"
                    ):
                        from poorstar.hierarchy.state import DispatchObservation, HierarchyState

                        prior = HierarchyState.from_dict(trace["prior_control_state"], c)
                        dispatch = trace.get("scheduler_dispatch_observation")
                        with torch.no_grad():
                            prediction = model.confidence_prediction(
                                frame.policy_observation(),
                                None if memory_before is None else memory_before.to(model.device),
                                prior,
                                feedback=tuple(ExecutionFeedback(**f) for f in trace["feedback"]),
                                dispatch_observation=None
                                if dispatch is None
                                else DispatchObservation(**dispatch),
                                task_allocations=trace.get("scheduler_task_allocations"),
                            )
                        measurement_sink.confidence(frame, session.control_state, float(prediction))
                    measurement_sink.control(
                        frame, session.control_state, session.audit_trace, "scheduler"
                    )
                if i == 0 and drafts[0] is not None:
                    finalize(
                        raws[0],
                        next_value=session.audit_trace["value"],
                        next_manager_value=session.audit_trace["manager_value"],
                    )
                drafts[i] = dict(
                    frame=frame,
                    body=None,
                    trace=deepcopy(session.audit_trace),
                    results=(),
                    memory_before=memory_before,
                )
                due[i] = now + delay
                phases[i] = "body" if decision == "ACT" else "scheduler"
                if decision == "WAIT" or delay:
                    action_prefixes[i].append(
                        dict(
                            phase="scheduler",
                            loop=now,
                            decision=decision,
                            delay=delay,
                            control=session.audit_trace["scheduler_control"],
                        )
                    )
                    continue
            if phases[i] == "body":
                budget()
                fresh = backend.controllers[i].observe()
                if int(fresh.observation.game_loop) != now or fresh.player_result:
                    raise ValueError("joint event body clock changed")
                q = query_pb2.RequestQuery()
                for tag in sorted(
                    int(u.tag) for u in fresh.observation.raw_data.units if u.alliance == 1
                ):
                    q.abilities.add(unit_tag=tag)
                budget()
                r = backend.controllers[i].query(q)
                frame = ActionFrame.from_protocol(
                    fresh,
                    backend.infos[i],
                    profile.catalog,
                    BoundQuery.bind(fresh.observation, q, r),
                    context=contexts[i],
                    episode=episode,
                )
                observe_goals(i, frame)
                machine = coarse_sample(
                    frame,
                    lambda h, o, c: session.choose_body(frame.policy_observation(), h, o, c),
                    profile,
                )
                if i == 0 and measurement_sink is not None:
                    measurement_sink.control(
                        frame, session.control_state, session.audit_trace, "body"
                    )
                prepared = PreparedBody(
                    drafts[i]["frame"].key,
                    frame,
                    now - drafts[i]["frame"].key.loop,
                    requested_delay=session.audit_trace["delay"],
                )
                budget()
                receipt = dispatch_live_bundle(backend.controllers[i], prepared, machine)
                receipts.append(
                    dict(
                        player_id=ids[i],
                        loop=now,
                        commands=receipt.command_count,
                        accepted=receipt.accepted_count,
                        results=receipt.results,
                    )
                )
                if model.hierarchy_config.common_receipts_mode == "shared_v1":
                    from hashlib import sha256

                    from poorstar.hierarchy.state import DispatchObservation
                    from poorstar.training.contracts import digest

                    evidence = digest(
                        [
                            episode,
                            ids[i],
                            now,
                            sha256(receipt.request_wire).hexdigest(),
                            sha256(receipt.response_wire).hexdigest(),
                        ]
                    )
                    session.submit_dispatch_observation(
                        DispatchObservation(
                            now,
                            (
                                receipt.command_count,
                                receipt.accepted_count,
                                receipt.command_count - receipt.accepted_count,
                                receipt.accepted_count,
                            ),
                            evidence,
                        )
                    )
                drafts[i].update(
                    body=frame, trace=deepcopy(session.audit_trace), results=receipt.results
                )
                action_prefixes[i].append(
                    dict(
                        phase="body",
                        loop=now,
                        tokens=deepcopy(session.audit_trace["body_tokens"]),
                        scheduler_control=session.audit_trace["scheduler_control"],
                        body_control=session.audit_trace["body_control"],
                    )
                )
                task_id = session.audit_trace["body_execution_task"]
                if task_id is not None:
                    from hashlib import sha256

                    from poorstar.training.contracts import digest

                    evidence = digest(
                        [
                            episode,
                            ids[i],
                            now,
                            task_id,
                            frame.key.__dict__,
                            sha256(receipt.request_wire).hexdigest(),
                            sha256(receipt.response_wire).hexdigest(),
                        ]
                    )
                    session.submit_feedback(
                        ExecutionFeedback(
                            evidence + "-dispatch", task_id, "dispatched", now, evidence
                        )
                    )
                    # One bundle owns one explicitly sampled task association;
                    # mixed acceptance cannot fabricate completion or success.
                    status = (
                        "accepted"
                        if receipt.all_accepted
                        else "rejected"
                        if receipt.accepted_count == 0
                        else None
                    )
                    if status:
                        session.submit_feedback(
                            ExecutionFeedback(
                                evidence + "-" + status, task_id, status, now, evidence
                            )
                        )
                due[i] = now + 1
                phases[i] = "scheduler"
        target = min(*due, maximum_loops, next_measurement)
        if target <= now:
            raise ValueError("joint actor scheduling made no progress")
        budget()
        backend.advance(target - now)
        budget()
        raws = backend.observe()
    sequence = RolloutSequence(tuple(steps), identity.sha256)
    sequence.validate(identity, policy_versions[0], 0, learning=False)
    if parameter_hashes != tuple(parameter_digest(m.state_dict()) for m in models):
        raise ValueError("actor parameters changed during immutable episode")
    return sequence, dict(
        schema="poorstar_v3_two_player_actor_report_v1",
        episode_id=episode,
        environment_loops=observed_loops,
        receipts=receipts,
        terminated=steps[-1].terminated,
        truncated=steps[-1].truncated,
        reward=steps[-1].reward,
        provenance=provenance,
        maximum_resident_steps=resident_limit,
        resident_steps=len(steps),
        boundary_reason=steps[-1].boundary_reason,
        checkpoint_sha256=behavior_hash,
        goal_observations=goal_observations,
        registered_interventions=interventions,
        initial_inputs=initial_inputs,
        deterministic_players=list(deterministic_players),
        capability_evaluated=False,
    )
