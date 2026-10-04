"""Tests: test training system actor."""

from copy import deepcopy

import pytest
import torch
from s2clientprotocol import query_pb2
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.training.actor import collect_pair
from tests.test_coarse_bc import setup
from tests.test_training_system_foundation import system


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ProtocolPair:
    """A declared event-clock/receipt fixture, with no game simulation."""

    version = "protocol-fixture"

    def __init__(self, terminal_loop=None):
        self.loop = 0
        self.terminal_loop = terminal_loop
        self.advances = []
        self.dispatches = []
        self.controllers = tuple(ProtocolController(self, pid) for pid in (1, 2))
        _, _, catalog, info = setup()
        self.catalogs = (catalog, deepcopy(catalog))
        self.infos = (info, deepcopy(info))

    def observe(self):
        return tuple(c.observe() for c in self.controllers)

    def advance(self, loops):
        assert loops > 0
        self.advances.append((self.loop, loops))
        self.loop += loops


class ProtocolController:
    def __init__(self, pair, player):
        self.pair = pair
        self.player = player

    def observe(self):
        f, *_ = setup(loop=self.pair.loop)
        raw = sc.ResponseObservation()
        raw.observation.ParseFromString(f._observation_wire)
        raw.observation.player_common.player_id = self.player
        if self.pair.terminal_loop is not None and self.pair.loop >= self.pair.terminal_loop:
            raw.player_result.add(
                player_id=self.player, result=sc.Victory if self.player == 1 else sc.Defeat
            )
        return raw

    def query(self, request):
        f, *_ = setup(loop=self.pair.loop)
        assert tuple(a.unit_tag for a in request.abilities) == f.slot_tags
        response = query_pb2.ResponseQuery()
        response.ParseFromString(f._query_response_wire)
        return response

    def actions(self, request):
        self.pair.dispatches.append((self.player, self.pair.loop, request.SerializeToString()))
        return sc.ResponseAction(result=[1] * len(request.actions))


def policies(delay, opponent_wait=True):
    learner, profile, _ = system("persistent_intent")
    opponent = deepcopy(learner.model)
    with torch.no_grad():
        learner.model.delay_head.bias.fill_(-30.0)
        learner.model.delay_head.bias[delay] = 30.0
        if opponent_wait:
            opponent.act_head.bias.copy_(torch.tensor([-30.0, 30.0]))
            opponent.delay_head.bias.fill_(-30.0)
            opponent.delay_head.bias[1] = 30.0
    return learner, opponent, profile


def replay_every_step(learner, sequence):
    hidden = None
    for step in sequence.steps:
        score = learner.scorer(step, hidden)
        hidden = score.next_hidden
        assert float(score.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
        assert float(score.value) == pytest.approx(step.old_value, abs=1e-6)


def test_joint_event_clock_advances_other_player_during_delayed_body():
    learner, opponent, profile = policies(2)
    backend = ProtocolPair(terminal_loop=3)
    sequence, report = collect_pair(
        backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=8,
        episode_id="joint-event-fixture",
    )
    assert backend.advances == [(0, 1), (1, 1), (2, 1)]
    assert [(p, t) for p, t, _ in backend.dispatches] == [(1, 2)]
    assert len(sequence.steps) == 1 and sequence.steps[0].payload.body_frame.key.loop == 2
    assert report["terminated"] and report["reward"] == 1.0 and report["environment_loops"] == 3
    assert not report["capability_evaluated"]
    replay_every_step(learner, sequence)


def test_budget_truncation_interrupts_body_without_fake_wait_or_outcome():
    learner, opponent, profile = policies(3)
    backend = ProtocolPair()
    sequence, report = collect_pair(
        backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=2,
        episode_id="joint-truncation-fixture",
    )
    step = sequence.steps[-1]
    assert not backend.dispatches and step.payload.trace["decision"] == "ACT"
    assert step.payload.interrupted_before_body and step.payload.body_frame is None
    assert step.truncated and not step.terminated and step.reward == 0 and step.reward_loop is None
    assert step.next_value != 0 and report["environment_loops"] == 2
    replay_every_step(learner, sequence)


def test_repeated_dispatch_receipts_have_distinct_causal_ids_and_replay():
    learner, opponent, profile = policies(0, opponent_wait=False)
    # Select a task consistently so repeated actual receipt ingestion is covered.
    with torch.no_grad():
        for model in (learner.model, opponent):
            model.task_operation_head.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))
            model.task_idle_embedding.fill_(-10.0)
            model.task_execution_query.weight.zero_()
            model.task_execution_query.bias.fill_(1.0)
    backend = ProtocolPair(terminal_loop=4)
    sequence, report = collect_pair(
        backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=6,
        episode_id="joint-receipts-fixture",
    )
    feedback = [f for step in sequence.steps for f in step.payload.trace["feedback"]]
    assert feedback and len({f["event_id"] for f in feedback}) == len(feedback)
    assert any(f["status"] == "accepted" for f in feedback)
    assert report["terminated"] and len(sequence.steps) == 4
    replay_every_step(learner, sequence)


def test_resident_step_cap_is_prospective_truncation_with_visible_bootstrap():
    learner, opponent, profile = policies(0)
    backend = ProtocolPair(terminal_loop=10)
    sequence, report = collect_pair(
        backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=12,
        maximum_resident_steps=2,
        episode_id="resident-bound-fixture",
    )
    assert len(sequence.steps) == report["resident_steps"] == report["maximum_resident_steps"] == 2
    assert report["environment_loops"] == 2 and not report["terminated"] and report["truncated"]
    assert report["boundary_reason"] == "resident_step_cap" and report["reward"] == 0
    assert sequence.steps[-1].next_value != 0
    replay_every_step(learner, sequence)
