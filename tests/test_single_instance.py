"""Tests for the Windows mutex wrapper without touching a real mutex."""

from asr_input.single_instance import ERROR_ALREADY_EXISTS, SingleInstance


def test_first_instance_keeps_handle_until_closed():
    closed = []
    instance = SingleInstance(
        "test",
        create_mutex=lambda name: 42,
        get_last_error=lambda: 0,
        close_handle=closed.append,
    )

    assert instance.acquired is True
    assert closed == []

    instance.close()
    instance.close()
    assert closed == [42]


def test_second_instance_closes_duplicate_handle_immediately():
    closed = []
    instance = SingleInstance(
        "test",
        create_mutex=lambda name: 84,
        get_last_error=lambda: ERROR_ALREADY_EXISTS,
        close_handle=closed.append,
    )

    assert instance.acquired is False
    assert closed == [84]


def test_context_manager_closes_owned_handle():
    closed = []
    with SingleInstance(
        "test",
        create_mutex=lambda name: 21,
        get_last_error=lambda: 0,
        close_handle=closed.append,
    ) as instance:
        assert instance.acquired is True

    assert closed == [21]
