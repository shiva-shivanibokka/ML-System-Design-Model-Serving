"""
Request Router.

Implements all traffic routing logic for the deployment state machine:

SHADOW mode:
  Every request → v1 (returned to user) + v2 (run silently in background).
  v2 result is NOT returned to user. It's logged for disagreement monitoring.
  Zero risk to users. Full visibility into v2 behaviour.

CANARY mode (5%, 25%, 50%):
  Weighted random routing: roll a float in [0,1].
  If roll < v2_fraction → route to v2.
  Otherwise → route to v1.
  If v2 raises an error or the circuit is open → fall back to v1.

FULL mode:
  All requests → v2.
  v1 kept warm but idle.
  If v2 circuit opens → fall back to v1 immediately.

ROLLED_BACK mode:
  All requests → v1.
  v2 receives no traffic (same as shadow but v2 also not run silently).
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

import structlog

from deployment.circuit_breaker import CircuitBreakerOpenError, circuit_breaker
from deployment.state_machine import DeploymentState, state_machine
from models.base import PredictionResult

log = structlog.get_logger(__name__)


class RequestRouter:
    """
    Routes inference requests to v1, v2, or both (shadow mode).
    Works with the DeploymentStateMachine and CircuitBreaker singletons.
    """

    def __init__(self) -> None:
        # Injected by FastAPI lifespan after models are loaded
        self._model_v1 = None
        self._model_v2 = None

    def set_models(self, model_v1, model_v2) -> None:
        self._model_v1 = model_v1
        self._model_v2 = model_v2

    async def route(
        self,
        text: str,
        trace_id: str,
    ) -> tuple[PredictionResult, Optional[PredictionResult], str]:
        """
        Route a request based on current deployment state.

        Returns:
            (user_result, shadow_v2_result, model_used)
            user_result      — the prediction returned to the user
            shadow_v2_result — v2 result if in shadow mode (else None)
            model_used       — "v1" | "v2" | "v1_cb_fallback" | "v1_error_fallback"
        """
        state = state_machine.state

        if state == DeploymentState.SHADOW:
            return await self._route_shadow(text, trace_id)

        elif state in (
            DeploymentState.CANARY_5,
            DeploymentState.CANARY_25,
            DeploymentState.CANARY_50,
        ):
            return await self._route_canary(text, trace_id)

        elif state == DeploymentState.FULL:
            return await self._route_full(text, trace_id)

        else:
            # ROLLED_BACK or unknown — all traffic to v1
            return await self._route_v1_only(text, trace_id)

    # -----------------------------------------------------------------------
    # Routing strategies
    # -----------------------------------------------------------------------

    async def _route_shadow(
        self, text: str, trace_id: str
    ) -> tuple[PredictionResult, Optional[PredictionResult], str]:
        """
        Shadow mode: v1 is primary, v2 runs silently in background.

        v2 is dispatched as a non-blocking asyncio task AFTER v1 responds.
        The user never waits for v2. If v2 fails, it's logged silently.
        """
        # v1 is the primary — user waits only for this
        v1_result = await asyncio.get_event_loop().run_in_executor(
            None, self._model_v1.predict, text
        )
        state_machine.record_request("v1", v1_result.latency_ms, error=False)

        # Fire v2 as a background task — user response already computed
        v2_result: Optional[PredictionResult] = None
        try:
            v2_result = await asyncio.get_event_loop().run_in_executor(
                None, self._safe_v2_predict, text
            )
            if v2_result:
                state_machine.record_request("v2", v2_result.latency_ms, error=False)
        except Exception as e:
            log.debug("shadow_v2_error", error=str(e), trace_id=trace_id)

        return v1_result, v2_result, "v1"

    async def _route_canary(
        self, text: str, trace_id: str
    ) -> tuple[PredictionResult, Optional[PredictionResult], str]:
        """
        Canary mode: weighted random routing.
        If v2 fails (circuit open or exception) → fall back to v1.
        """
        v2_fraction = state_machine.v2_traffic_fraction
        use_v2 = random.random() < v2_fraction

        if use_v2:
            # Try v2
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._safe_v2_predict_or_raise, text
                )
                state_machine.record_request("v2", result.latency_ms, error=False)
                return result, None, "v2"
            except CircuitBreakerOpenError as e:
                log.info(
                    "canary_cb_fallback",
                    trace_id=trace_id,
                    reason=str(e),
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._model_v1.predict, text
                )
                state_machine.record_request("v1", result.latency_ms, error=False)
                return result, None, "v1_cb_fallback"
            except Exception as e:
                log.warning(
                    "canary_v2_error_fallback",
                    trace_id=trace_id,
                    error=str(e),
                )
                state_machine.record_request("v2", 0.0, error=True)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._model_v1.predict, text
                )
                state_machine.record_request("v1", result.latency_ms, error=False)
                return result, None, "v1_error_fallback"
        else:
            # v1 slice
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._model_v1.predict, text
            )
            state_machine.record_request("v1", result.latency_ms, error=False)
            return result, None, "v1"

    async def _route_full(
        self, text: str, trace_id: str
    ) -> tuple[PredictionResult, Optional[PredictionResult], str]:
        """
        Full deployment: all traffic to v2.
        If circuit is open → fall back to v1 (Blue/Green instant switch).
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._safe_v2_predict_or_raise, text
            )
            state_machine.record_request("v2", result.latency_ms, error=False)
            return result, None, "v2"
        except CircuitBreakerOpenError as e:
            log.warning(
                "full_deployment_cb_fallback",
                trace_id=trace_id,
                reason=str(e),
            )
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._model_v1.predict, text
            )
            state_machine.record_request("v1", result.latency_ms, error=False)
            return result, None, "v1_cb_fallback"
        except Exception as e:
            log.warning("full_v2_error_fallback", trace_id=trace_id, error=str(e))
            state_machine.record_request("v2", 0.0, error=True)
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._model_v1.predict, text
            )
            state_machine.record_request("v1", result.latency_ms, error=False)
            return result, None, "v1_error_fallback"

    async def _route_v1_only(
        self, text: str, trace_id: str
    ) -> tuple[PredictionResult, Optional[PredictionResult], str]:
        """Rolled back / unknown state — pure v1."""
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._model_v1.predict, text
        )
        state_machine.record_request("v1", result.latency_ms, error=False)
        return result, None, "v1"

    # -----------------------------------------------------------------------
    # v2 inference wrappers
    # -----------------------------------------------------------------------

    def _safe_v2_predict(self, text: str) -> Optional[PredictionResult]:
        """
        Run v2 through circuit breaker. Returns None on any error.
        Used in shadow mode where errors are silently absorbed.
        """
        try:
            return circuit_breaker.call(self._model_v2.predict, text)
        except Exception:
            return None

    def _safe_v2_predict_or_raise(self, text: str) -> PredictionResult:
        """
        Run v2 through circuit breaker. Raises on error.
        Used in canary/full modes where caller handles the exception.
        """
        return circuit_breaker.call(self._model_v2.predict, text)


# Module-level singleton
router = RequestRouter()
