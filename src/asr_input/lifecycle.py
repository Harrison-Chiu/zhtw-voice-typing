"""Pure lifecycle state machine for the tray application.

The state machine owns product decisions but performs no I/O.  Callers execute
the returned effects and report completion back with the matching generation.
Keeping this layer pure makes load/unload races and simultaneous recording / ASR
work testable without a microphone, CUDA, or a Windows tray.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace


class ModelState(enum.Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


class CaptureState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    ERROR = "error"


class EffectKind(enum.Enum):
    START_CAPTURE = "start_capture"
    STOP_CAPTURE = "stop_capture"
    LOAD_MODEL = "load_model"
    UNLOAD_MODEL = "unload_model"
    TRANSCRIBE_NEXT = "transcribe_next"
    SCHEDULE_IDLE_UNLOAD = "schedule_idle_unload"
    CANCEL_IDLE_UNLOAD = "cancel_idle_unload"


@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    generation: int | None = None


@dataclass(frozen=True)
class LifecycleSnapshot:
    model: ModelState = ModelState.UNLOADED
    capture: CaptureState = CaptureState.IDLE
    queued_jobs: int = 0
    transcribing: bool = False
    model_generation: int = 0
    idle_generation: int = 0
    unload_when_idle: bool = False
    error: str | None = None

    @property
    def busy(self) -> bool:
        return self.capture is CaptureState.RECORDING or self.queued_jobs > 0 or self.transcribing


class HotkeyLatch:
    """Emit once per physical chord press, regardless of key-repeat events."""

    def __init__(self) -> None:
        self._latched = False

    def update(self, chord_pressed: bool) -> bool:
        if not chord_pressed:
            self._latched = False
            return False
        if self._latched:
            return False
        self._latched = True
        return True


class AudioQueueBudget:
    """Duration-based memory safety net; rejection never deletes persisted work."""

    def __init__(self, max_seconds: float, warning_ratio: float = 0.8) -> None:
        self.max_seconds = max_seconds
        self.warning_ratio = warning_ratio
        self.current_seconds = 0.0

    def try_add(self, duration_sec: float) -> tuple[bool, bool]:
        projected = self.current_seconds + max(0.0, duration_sec)
        if projected > self.max_seconds:
            return False, True
        self.current_seconds = projected
        return True, projected >= self.max_seconds * self.warning_ratio

    def remove(self, duration_sec: float) -> None:
        self.current_seconds = max(0.0, self.current_seconds - max(0.0, duration_sec))


class LifecycleMachine:
    """Synchronous reducer for model, capture, queue, and idle-unload state."""

    def __init__(self) -> None:
        self._snapshot = LifecycleSnapshot()

    @property
    def snapshot(self) -> LifecycleSnapshot:
        return self._snapshot

    def begin_capture(self) -> tuple[Effect, ...]:
        if self._snapshot.capture is CaptureState.RECORDING:
            return ()

        effects = [Effect(EffectKind.CANCEL_IDLE_UNLOAD)]
        self._invalidate_idle_timer()
        self._snapshot = replace(
            self._snapshot,
            capture=CaptureState.RECORDING,
            error=None,
        )
        effects.append(Effect(EffectKind.START_CAPTURE))

        if self._snapshot.model in (ModelState.UNLOADED, ModelState.ERROR):
            generation = self._snapshot.model_generation + 1
            self._snapshot = replace(
                self._snapshot,
                model=ModelState.LOADING,
                model_generation=generation,
            )
            effects.append(Effect(EffectKind.LOAD_MODEL, generation))

        return tuple(effects)

    def end_capture(self) -> tuple[Effect, ...]:
        if self._snapshot.capture is not CaptureState.RECORDING:
            return ()
        self._snapshot = replace(self._snapshot, capture=CaptureState.IDLE)
        return (Effect(EffectKind.STOP_CAPTURE),)

    def capture_failed(self, message: str) -> None:
        self._snapshot = replace(
            self._snapshot,
            capture=CaptureState.ERROR,
            error=message,
        )

    def clear_capture_error(self) -> None:
        if self._snapshot.capture is CaptureState.ERROR:
            self._snapshot = replace(
                self._snapshot,
                capture=CaptureState.IDLE,
                error=None,
            )

    def queue_job(self) -> tuple[Effect, ...]:
        self._invalidate_idle_timer()
        self._snapshot = replace(
            self._snapshot,
            queued_jobs=self._snapshot.queued_jobs + 1,
            error=None,
        )
        effects = [Effect(EffectKind.CANCEL_IDLE_UNLOAD)]
        if self._snapshot.model in (ModelState.UNLOADED, ModelState.ERROR):
            generation = self._snapshot.model_generation + 1
            self._snapshot = replace(
                self._snapshot,
                model=ModelState.LOADING,
                model_generation=generation,
                error=None,
            )
            effects.append(Effect(EffectKind.LOAD_MODEL, generation))
        if self._can_start_job():
            effects.append(self._start_next_job())
        return tuple(effects)

    def model_ready(self, generation: int) -> tuple[Effect, ...]:
        if generation != self._snapshot.model_generation:
            return ()
        if self._snapshot.model is not ModelState.LOADING:
            return ()

        self._snapshot = replace(self._snapshot, model=ModelState.READY, error=None)
        if self._can_start_job():
            return (self._start_next_job(),)
        if self._snapshot.unload_when_idle and not self._snapshot.busy:
            return self._begin_unload()
        return self._schedule_idle_if_possible()

    def model_failed(self, generation: int, message: str) -> None:
        if generation != self._snapshot.model_generation:
            return
        if self._snapshot.model is not ModelState.LOADING:
            return
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.ERROR,
            transcribing=False,
            error=message,
        )

    def retry_model_load(self) -> tuple[Effect, ...]:
        if self._snapshot.model is not ModelState.ERROR:
            return ()
        generation = self._snapshot.model_generation + 1
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.LOADING,
            model_generation=generation,
            error=None,
        )
        return (Effect(EffectKind.LOAD_MODEL, generation),)

    def request_model_load(self) -> tuple[Effect, ...]:
        if self._snapshot.model is ModelState.ERROR:
            return self.retry_model_load()
        if self._snapshot.model is not ModelState.UNLOADED:
            return ()
        generation = self._snapshot.model_generation + 1
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.LOADING,
            model_generation=generation,
            error=None,
        )
        return (Effect(EffectKind.LOAD_MODEL, generation),)

    def job_completed(self) -> tuple[Effect, ...]:
        if not self._snapshot.transcribing:
            return ()
        self._snapshot = replace(self._snapshot, transcribing=False, error=None)

        if self._can_start_job():
            return (self._start_next_job(),)
        if self._snapshot.unload_when_idle:
            return self._begin_unload()
        return self._schedule_idle_if_possible()

    def job_failed(self, message: str) -> tuple[Effect, ...]:
        if not self._snapshot.transcribing:
            return ()
        self._snapshot = replace(
            self._snapshot,
            transcribing=False,
            error=message,
        )
        if self._can_start_job():
            return (self._start_next_job(),)
        return self._schedule_idle_if_possible()

    def request_unload(self) -> tuple[Effect, ...]:
        if self._snapshot.model not in (ModelState.READY, ModelState.LOADING):
            return ()
        if self._snapshot.busy or self._snapshot.model is ModelState.LOADING:
            self._snapshot = replace(self._snapshot, unload_when_idle=True)
            return ()
        return self._begin_unload()

    def idle_expired(self, generation: int) -> tuple[Effect, ...]:
        if generation != self._snapshot.idle_generation:
            return ()
        if self._snapshot.model is not ModelState.READY or self._snapshot.busy:
            return ()
        return self._begin_unload()

    def reschedule_idle_unload(self) -> tuple[Effect, ...]:
        self._invalidate_idle_timer()
        return (Effect(EffectKind.CANCEL_IDLE_UNLOAD),) + self._schedule_idle_if_possible()

    def model_unloaded(self, generation: int) -> tuple[Effect, ...]:
        if generation != self._snapshot.model_generation:
            return ()
        if self._snapshot.model is not ModelState.UNLOADING:
            return ()
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.UNLOADED,
            unload_when_idle=False,
        )
        if self._snapshot.queued_jobs:
            return self.request_model_load()
        return ()

    def model_unload_failed(self, generation: int, message: str) -> None:
        if generation != self._snapshot.model_generation:
            return
        if self._snapshot.model is not ModelState.UNLOADING:
            return
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.ERROR,
            unload_when_idle=False,
            error=message,
        )

    def _can_start_job(self) -> bool:
        return (
            self._snapshot.model is ModelState.READY
            and self._snapshot.queued_jobs > 0
            and not self._snapshot.transcribing
        )

    def _start_next_job(self) -> Effect:
        self._snapshot = replace(
            self._snapshot,
            queued_jobs=self._snapshot.queued_jobs - 1,
            transcribing=True,
        )
        return Effect(EffectKind.TRANSCRIBE_NEXT)

    def _schedule_idle_if_possible(self) -> tuple[Effect, ...]:
        if self._snapshot.model is not ModelState.READY or self._snapshot.busy:
            return ()
        generation = self._snapshot.idle_generation + 1
        self._snapshot = replace(self._snapshot, idle_generation=generation)
        return (Effect(EffectKind.SCHEDULE_IDLE_UNLOAD, generation),)

    def _invalidate_idle_timer(self) -> None:
        self._snapshot = replace(
            self._snapshot,
            idle_generation=self._snapshot.idle_generation + 1,
        )

    def _begin_unload(self) -> tuple[Effect, ...]:
        generation = self._snapshot.model_generation + 1
        self._snapshot = replace(
            self._snapshot,
            model=ModelState.UNLOADING,
            model_generation=generation,
            unload_when_idle=False,
        )
        return (Effect(EffectKind.UNLOAD_MODEL, generation),)
