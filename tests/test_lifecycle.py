"""Lifecycle tests that do not require CUDA, a microphone, or Windows UI."""

from asr_input.lifecycle import (
    AudioQueueBudget,
    CaptureState,
    EffectKind,
    HotkeyLatch,
    LifecycleMachine,
    ModelState,
)


def kinds(effects):
    return [effect.kind for effect in effects]


def test_hotkey_latch_emits_once_until_chord_is_released():
    latch = HotkeyLatch()

    assert latch.update(True) is True
    assert latch.update(True) is False
    assert latch.update(True) is False
    assert latch.update(False) is False
    assert latch.update(True) is True


def test_audio_queue_budget_warns_near_limit_and_rejects_without_counting():
    budget = AudioQueueBudget(max_seconds=100, warning_ratio=0.8)

    assert budget.try_add(79) == (True, False)
    assert budget.try_add(2) == (True, True)
    assert budget.try_add(20) == (False, True)
    assert budget.current_seconds == 81

    budget.remove(40)
    assert budget.current_seconds == 41


def test_capture_starts_before_model_load_and_does_not_duplicate_load():
    machine = LifecycleMachine()

    first = machine.begin_capture()
    assert kinds(first) == [
        EffectKind.CANCEL_IDLE_UNLOAD,
        EffectKind.START_CAPTURE,
        EffectKind.LOAD_MODEL,
    ]
    assert machine.snapshot.capture is CaptureState.RECORDING
    assert machine.snapshot.model is ModelState.LOADING

    assert machine.end_capture()[0].kind is EffectKind.STOP_CAPTURE
    machine.queue_job()

    second = machine.begin_capture()
    assert kinds(second) == [EffectKind.CANCEL_IDLE_UNLOAD, EffectKind.START_CAPTURE]
    assert machine.snapshot.queued_jobs == 1


def test_ready_model_drains_jobs_one_at_a_time_while_capture_can_continue():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()
    machine.queue_job()
    machine.begin_capture()
    machine.end_capture()
    machine.queue_job()

    effects = machine.model_ready(load.generation)
    assert kinds(effects) == [EffectKind.TRANSCRIBE_NEXT]
    assert machine.snapshot.transcribing is True
    assert machine.snapshot.queued_jobs == 1

    effects = machine.job_completed()
    assert kinds(effects) == [EffectKind.TRANSCRIBE_NEXT]
    assert machine.snapshot.transcribing is True
    assert machine.snapshot.queued_jobs == 0

    effects = machine.job_completed()
    assert kinds(effects) == [EffectKind.SCHEDULE_IDLE_UNLOAD]
    assert machine.snapshot.transcribing is False


def test_new_or_successful_work_clears_previous_worker_error():
    machine = LifecycleMachine()
    load = machine.request_model_load()[0]
    machine.model_ready(load.generation)
    machine.queue_job()
    machine.job_failed("pipeline failed")
    assert machine.snapshot.error == "pipeline failed"

    machine.queue_job()
    assert machine.snapshot.error is None
    machine.job_completed()
    assert machine.snapshot.error is None


def test_stale_idle_timer_cannot_unload_after_new_capture():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()
    machine.model_ready(load.generation)
    scheduled = machine.snapshot.idle_generation

    machine.begin_capture()
    assert machine.idle_expired(scheduled) == ()
    assert machine.snapshot.model is ModelState.READY


def test_idle_timer_can_be_rescheduled_after_preference_change():
    machine = LifecycleMachine()
    load = machine.request_model_load()[0]
    first = machine.model_ready(load.generation)[0]

    effects = machine.reschedule_idle_unload()

    assert kinds(effects) == [
        EffectKind.CANCEL_IDLE_UNLOAD,
        EffectKind.SCHEDULE_IDLE_UNLOAD,
    ]
    assert effects[-1].generation != first.generation


def test_idle_unload_and_next_capture_reuse_the_same_load_path():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()
    idle_effect = machine.model_ready(load.generation)[0]

    unload_effect = machine.idle_expired(idle_effect.generation)[0]
    assert unload_effect.kind is EffectKind.UNLOAD_MODEL
    assert machine.model_unloaded(unload_effect.generation) == ()
    assert machine.snapshot.model is ModelState.UNLOADED

    effects = machine.begin_capture()
    assert kinds(effects)[-2:] == [EffectKind.START_CAPTURE, EffectKind.LOAD_MODEL]


def test_failed_load_keeps_jobs_and_retry_uses_new_generation():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()
    machine.queue_job()

    machine.model_failed(load.generation, "CUDA unavailable")
    assert machine.snapshot.model is ModelState.ERROR
    assert machine.snapshot.queued_jobs == 1

    retry = machine.retry_model_load()[0]
    assert retry.kind is EffectKind.LOAD_MODEL
    assert retry.generation != load.generation

    effects = machine.model_ready(retry.generation)
    assert kinds(effects) == [EffectKind.TRANSCRIBE_NEXT]


def test_model_can_be_loaded_from_the_tray_without_starting_capture():
    machine = LifecycleMachine()

    effect = machine.request_model_load()[0]
    assert effect.kind is EffectKind.LOAD_MODEL
    assert machine.snapshot.capture is CaptureState.IDLE
    assert machine.snapshot.model is ModelState.LOADING


def test_manual_unload_waits_until_last_job_finishes():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()
    machine.queue_job()
    machine.model_ready(load.generation)

    assert machine.request_unload() == ()
    assert machine.snapshot.unload_when_idle is True

    effects = machine.job_completed()
    assert kinds(effects) == [EffectKind.UNLOAD_MODEL]
    assert machine.snapshot.model is ModelState.UNLOADING


def test_unload_requested_during_load_runs_as_soon_as_ready_when_idle():
    machine = LifecycleMachine()
    load = machine.begin_capture()[-1]
    machine.end_capture()

    assert machine.request_unload() == ()
    assert machine.snapshot.unload_when_idle is True

    effects = machine.model_ready(load.generation)
    assert kinds(effects) == [EffectKind.UNLOAD_MODEL]
    assert machine.snapshot.model is ModelState.UNLOADING


def test_job_arriving_during_unload_reloads_after_unload_completes():
    machine = LifecycleMachine()
    load = machine.request_model_load()[0]
    idle = machine.model_ready(load.generation)[0]
    unload = machine.idle_expired(idle.generation)[0]

    effects = machine.queue_job()
    assert kinds(effects) == [EffectKind.CANCEL_IDLE_UNLOAD]
    assert machine.snapshot.queued_jobs == 1

    effects = machine.model_unloaded(unload.generation)
    assert kinds(effects) == [EffectKind.LOAD_MODEL]
    assert machine.snapshot.model is ModelState.LOADING


def test_unload_failure_becomes_retryable_error():
    machine = LifecycleMachine()
    load = machine.request_model_load()[0]
    idle = machine.model_ready(load.generation)[0]
    unload = machine.idle_expired(idle.generation)[0]

    machine.model_unload_failed(unload.generation, "release failed")

    assert machine.snapshot.model is ModelState.ERROR
    assert machine.retry_model_load()[0].kind is EffectKind.LOAD_MODEL
