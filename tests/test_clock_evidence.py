from poorstar.engine.clock_evidence import reconcile


def event(loop=4, mode="None", target=None, user=0, flags=0):
    return dict(
        _event="NNet.Game.SCmdEvent",
        _gameloop=loop,
        _userid={"m_userId": user},
        m_data={mode: target},
        m_cmdFlags=flags,
        m_abil=None,
    )


def command(loop=5, point=None, tag=None, queue=False):
    return dict(
        kind="unit_command",
        response_loop=loop,
        position=point,
        target_tag=tag,
        queue=queue,
        protocol_game_loop=None,
    )


def test_next_frame_content_without_action_timestamp():
    result = reconcile(
        [event(mode="TargetPoint", target={"x": 4096, "y": 8192}, flags=2)],
        [command(point=[1, 2], queue=True)],
        user_id=0,
    )
    assert result["counts"]["direct_unique_position"] == 1
    assert result["counts"]["engine_missing_action_game_loop"] == 1
    assert not result["timestamp_alignment_verified"]


def test_multiple_identical_commands_never_ordinal_joined():
    result = reconcile([event(), event()], [command(), command()], user_id=0)
    assert result["counts"]["direct_ambiguous_content_candidate"] == 2
    assert "direct_unique_content_candidate" not in result["counts"]


def test_other_player_and_manager_states_are_explicit():
    manager = dict(
        _event="NNet.Game.SCommandManagerStateEvent",
        _gameloop=4,
        _userid={"m_userId": 0},
        m_state=1,
    )
    result = reconcile([event(user=1), manager, dict(manager, m_state=0)], [command()], user_id=0)
    assert result["counts"]["candidate_event_commands"] == 1
    assert result["counts"]["exact_frame_multiplicity"] == 1


def test_missing_entity_command_keeps_zero_target_denominator():
    result = reconcile([event(mode="TargetUnit", target={"m_tag": 0})], [], user_id=0)
    assert result["counts"]["direct_without_content_candidate"] == 1
    assert result["unmatched_direct_commands"][0]["original_target_tag_zero"]
    assert len(result["frame_multiplicity_mismatches"]) == 1


def test_opaque_data_is_retained_without_guessing_no_target_content_identity():
    result = reconcile([event(mode="Data", target=7654321)], [command()], user_id=0)
    assert (
        result["counts"]["candidate_event_commands"]
        == result["counts"]["engine_unit_commands"]
        == 1
    )
    assert result["counts"]["exact_frame_multiplicity"] == 1
    assert result["counts"]["direct_mode_opaque_data"] == 1
    assert result["counts"]["direct_without_content_candidate"] == 1
    assert result["unmatched_direct_commands"][0]["opaque_data_argument_unresolved"]
    assert "direct_unique_content_candidate" not in result["counts"]
    assert not result["native_data_target_semantics_verified"]
    assert not result["training_ready"] and not result["timestamp_alignment_verified"]
    assert "7654321" not in str(result)
