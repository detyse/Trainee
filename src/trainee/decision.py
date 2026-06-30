from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from trainee.llm import LLMClient, ProviderCallError, ProviderCompletion, ProviderDispatchError
from trainee.models import AgentDecision, AgentTrace, ProjectContext, ProjectSpec, PromptPreview
from trainee.prompt_assembler import PromptAssembler, PromptEnvelope
from trainee.prompt_documents import PromptDocument
from trainee.providers import provider_has_key, provider_model
from trainee.research_state import ResearchRoundState, ResearchState
from trainee.settings import Settings


# ProviderCallError and ProviderCompletion are imported here to preserve the
# historical `trainee.decision` import path while provider logic lives in llm.
@dataclass
class DecisionResult:
    decision: AgentDecision
    prompt_preview: Optional[PromptPreview] = None
    agent_trace: Optional[AgentTrace] = None


@dataclass(frozen=True)
class ParamAdjustment:
    name: str
    before: Any
    after: Any


class DecisionEngine:
    def __init__(
        self,
        settings: Settings,
        prompt_assembler: Optional[PromptAssembler] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.settings = settings
        self.prompt_assembler = prompt_assembler or PromptAssembler()
        self.llm_client = llm_client or LLMClient(settings)

    async def decide(
        self,
        *,
        spec: ProjectSpec,
        context: ProjectContext,
        research_state: ResearchState,
        current_params: dict[str, Any],
        prompt_documents: list[PromptDocument],
    ) -> AgentDecision:
        result = await self.decide_with_prompt(
            spec=spec,
            context=context,
            research_state=research_state,
            current_params=current_params,
            prompt_documents=prompt_documents,
        )
        return result.decision

    async def decide_with_prompt(
        self,
        *,
        spec: ProjectSpec,
        context: ProjectContext,
        research_state: ResearchState,
        current_params: dict[str, Any],
        prompt_documents: list[PromptDocument],
    ) -> DecisionResult:
        if not spec.tunable_params:
            reason = "No tunable_params are configured, so the loop stops after collecting the latest result."
            return DecisionResult(
                decision=AgentDecision(
                    action="stop",
                    next_params=current_params,
                    reason=reason,
                    focus_metrics=self._focus_metrics(spec),
                    latest_round_judgement=self._latest_judgement(research_state),
                    confidence=1.0,
                ),
                agent_trace=self._new_trace(status="not_called", fallback_reason=reason),
            )

        envelope = self.prompt_assembler.assemble(
            spec=spec,
            context=context,
            research_state=research_state,
            current_params=current_params,
            prompt_documents=prompt_documents,
            system_prompt=self.settings.system_prompt,
        )
        prompt_preview = self._build_prompt_preview(envelope, status="not_sent")
        providers = self.llm_client.configured_providers()
        if not providers:
            reason = "No configured LLM provider is available; stopping instead of using heuristic fallback."
            agent_trace = self._new_trace(
                status="not_called",
                fallback_reason=reason,
                provider_error=reason,
                force=True,
            )
            return DecisionResult(
                decision=self._stop_decision(spec, research_state, current_params, reason, confidence=0.0),
                prompt_preview=prompt_preview.model_copy(update={"status": "not_called"}),
                agent_trace=agent_trace,
            )

        try:
            dispatch = await self.llm_client.complete_with_fallback(envelope.system_prompt, envelope.user_prompt)
        except ProviderDispatchError as exc:
            attempts = exc.attempts
            last_attempt = attempts[-1] if attempts else None
            last_error = exc.last_error
            reason = "All configured LLM providers failed; stopping instead of using heuristic fallback."
            if attempts:
                reason += " " + "; ".join(
                    f"{attempt['provider']} {attempt['status']}"
                    + (f" HTTP {attempt['http_status']}" if attempt.get("http_status") is not None else "")
                    + (f": {attempt['error_message']}" if attempt.get("error_message") else "")
                    for attempt in attempts
                )
            if last_attempt:
                prompt_preview = self._build_prompt_preview(
                    envelope,
                    status=str(last_attempt["status"]),
                    provider=str(last_attempt["provider"]),
                )
                agent_trace = self._new_trace(
                    provider=str(last_attempt["provider"]),
                    model=str(last_attempt.get("model") or provider_model(self.settings, str(last_attempt["provider"]))),
                    status=str(last_attempt["status"]),
                    attempts=attempts,
                    http_status=last_error.http_status if last_error is not None else last_attempt.get("http_status"),
                    request_id=last_error.request_id if last_error is not None else last_attempt.get("request_id"),
                    raw_response_body=last_error.raw_response_body if last_error is not None else None,
                    error_body=last_error.error_body if last_error is not None else None,
                    provider_error=str(last_error) if last_error is not None else str(last_attempt.get("error_message") or reason),
                    fallback_reason=str(last_error) if last_error is not None else str(last_attempt.get("error_message") or reason),
                    force=True,
                )
            else:
                agent_trace = self._new_trace(
                    status="request_failed",
                    attempts=attempts,
                    provider_error=reason,
                    fallback_reason=reason,
                    force=True,
                )
            return DecisionResult(
                decision=self._stop_decision(spec, research_state, current_params, reason, confidence=0.0),
                prompt_preview=prompt_preview,
                agent_trace=agent_trace,
            )

        provider = dispatch.provider
        completion = dispatch.completion
        attempts = dispatch.attempts
        prompt_preview = self._build_prompt_preview(envelope, status="sent", provider=provider)
        has_failed_attempt = any(not attempt.get("ok") for attempt in attempts)
        if self.settings.agent_debug_enabled:
            agent_trace = self._new_trace(
                provider=provider,
                model=dispatch.model,
                status="success",
                attempts=attempts,
                http_status=completion.http_status,
                request_id=completion.request_id,
                raw_response_body=completion.raw_response_body,
                raw_output=completion.content,
                usage=completion.usage,
                finish_reason=completion.finish_reason,
            )
        elif has_failed_attempt:
            agent_trace = self._new_trace(
                provider=provider,
                model=dispatch.model,
                status="success",
                attempts=attempts,
                http_status=completion.http_status,
                request_id=completion.request_id,
                force=True,
            )
        else:
            agent_trace = None
        try:
            candidate = self._extract_json(completion.content)
        except (ValueError, json.JSONDecodeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if agent_trace is not None:
                agent_trace.status = "parse_failed"
                agent_trace.parse_error = message
                agent_trace.fallback_reason = message
            else:
                agent_trace = self._new_trace(
                    provider=provider,
                    model=dispatch.model,
                    status="parse_failed",
                    attempts=attempts,
                    parse_error=message,
                    fallback_reason=message,
                    force=True,
                )
            prompt_preview = prompt_preview.model_copy(update={"status": "parse_failed"})
            reason = f"LLM decision response could not be parsed; stopping instead of using heuristic fallback: {message}"
            return DecisionResult(
                decision=self._stop_decision(spec, research_state, current_params, reason, confidence=0.0),
                prompt_preview=prompt_preview,
                agent_trace=agent_trace,
            )

        if agent_trace is not None:
            agent_trace.extracted_json = candidate
        try:
            decision = AgentDecision.model_validate(candidate)
        except ValidationError as exc:
            message = str(exc)
            if agent_trace is not None:
                agent_trace.status = "validation_failed"
                agent_trace.validation_error = message
                agent_trace.fallback_reason = message
            else:
                agent_trace = self._new_trace(
                    provider=provider,
                    model=dispatch.model,
                    status="validation_failed",
                    attempts=attempts,
                    validation_error=message,
                    fallback_reason=message,
                    force=True,
                )
            prompt_preview = prompt_preview.model_copy(update={"status": "validation_failed"})
            reason = f"LLM decision response failed validation; stopping instead of using heuristic fallback: {message}"
            return DecisionResult(
                decision=self._stop_decision(spec, research_state, current_params, reason, confidence=0.0),
                prompt_preview=prompt_preview,
                agent_trace=agent_trace,
            )

        try:
            decision.next_params = spec.merge_param_values(decision.next_params, base=current_params)
        except ValueError as exc:
            message = f"tunable parameter validation failed: {exc}"
            if agent_trace is not None:
                agent_trace.status = "validation_failed"
                agent_trace.validation_error = message
                agent_trace.fallback_reason = message
            else:
                agent_trace = self._new_trace(
                    provider=provider,
                    model=dispatch.model,
                    status="validation_failed",
                    attempts=attempts,
                    validation_error=message,
                    fallback_reason=message,
                    force=True,
                )
            prompt_preview = prompt_preview.model_copy(update={"status": "validation_failed"})
            reason = f"LLM decision proposed invalid tunable params; stopping instead of using heuristic fallback: {message}"
            return DecisionResult(
                decision=self._stop_decision(spec, research_state, current_params, reason, confidence=0.0),
                prompt_preview=prompt_preview,
                agent_trace=agent_trace,
            )

        return DecisionResult(decision=decision, prompt_preview=prompt_preview, agent_trace=agent_trace)

    def build_prompt_preview(
        self,
        *,
        spec: ProjectSpec,
        context: ProjectContext,
        research_state: ResearchState,
        current_params: dict[str, Any],
        prompt_documents: list[PromptDocument],
        status: str = "preview",
    ) -> PromptPreview:
        envelope = self.prompt_assembler.assemble(
            spec=spec,
            context=context,
            research_state=research_state,
            current_params=current_params,
            prompt_documents=prompt_documents,
            system_prompt=self.settings.system_prompt,
        )
        return self._build_prompt_preview(envelope, status=status)

    async def probe(self, prompt: str, image: Optional[dict[str, str]] = None) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")

        provider = self._require_active_provider_configured()
        completion = await self.llm_client.complete_active(self._probe_system_prompt(), prompt, image=image)
        return {
            "provider": provider,
            "model": provider_model(self.settings, provider),
            "has_image": image is not None,
            "content": completion.content,
        }

    async def analyze_image(
        self,
        prompt: str,
        image: dict[str, str],
        *,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")
        system = system_prompt or self._visual_analysis_system_prompt()

        provider = self._require_active_provider_configured()
        completion = await self.llm_client.complete_active(system, prompt, image=image)
        return {
            "provider": provider,
            "model": provider_model(self.settings, provider),
            "has_image": True,
            "content": completion.content,
            "usage": completion.usage,
        }

    def _build_prompt_preview(self, envelope: PromptEnvelope, status: str, provider: Optional[str] = None) -> PromptPreview:
        provider = provider or self.settings.llm_provider
        model = provider_model(self.settings, provider)
        payload: dict[str, Any] = {
            "system": envelope.system_prompt,
            "user": envelope.user_prompt,
        }
        if provider != "none":
            payload = self.llm_client.build_payload(provider, envelope.system_prompt, envelope.user_prompt)
        return PromptPreview(
            provider=provider,
            model=model,
            status=status,
            system_prompt=envelope.system_prompt,
            user_prompt=envelope.user_prompt,
            payload=payload,
            static_context_json=envelope.static_context_json,
            dynamic_state_json=envelope.dynamic_state_json,
        )

    def _require_active_provider_configured(self) -> str:
        provider = self.settings.llm_provider
        if provider == "none":
            raise ValueError("LLM provider is disabled")
        if not provider_has_key(self.settings, provider):
            raise ValueError(self._api_key_config_message(provider))
        return provider

    def _api_key_config_message(self, provider: str) -> str:
        if provider == "openai":
            return "OPENAI_API_KEY is not configured"
        if provider == "moonshot":
            return "MOONSHOT_API_KEY is not configured"
        if provider == "anthropic":
            return "ANTHROPIC_API_KEY is not configured"
        return f"{provider.upper()} API key is not configured."

    def _provider_model(self, provider: str) -> Optional[str]:
        return provider_model(self.settings, provider)

    def _new_trace(
        self,
        *,
        status: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        force: bool = False,
        **updates: Any,
    ) -> Optional[AgentTrace]:
        if not force and not self.settings.agent_debug_enabled:
            return None
        return AgentTrace(
            provider=provider or self.settings.llm_provider,
            model=model if model is not None else self._provider_model(provider or self.settings.llm_provider),
            status=status,
            **updates,
        )

    def _probe_system_prompt(self) -> str:
        return "You are a concise API test assistant. Answer the user's prompt directly."

    def _visual_analysis_system_prompt(self) -> str:
        return (
            "You are a conservative visual analysis assistant for machine learning training diagnostics. "
            "Return compact JSON only. Do not recommend parameter changes directly; summarize visual evidence."
        )

    def _stop_decision(
        self,
        spec: ProjectSpec,
        research_state: ResearchState,
        current_params: dict[str, Any],
        reason: str,
        *,
        confidence: float = 0.0,
    ) -> AgentDecision:
        latest = research_state.latest_round
        return AgentDecision(
            action="stop",
            next_params=spec.merge_param_values(base=current_params),
            reason=reason,
            focus_metrics=self._focus_metrics(spec),
            latest_round_judgement=self._latest_judgement(research_state),
            compare_to_baseline=self._comparison_text("baseline", latest, research_state.baseline_round) if latest else "",
            compare_to_best=self._comparison_text("best", latest, research_state.best_so_far_round) if latest else "",
            avoid_repeating=research_state.rejected_change_signatures,
            confidence=confidence,
        )

    def _heuristic_decision(
        self,
        spec: ProjectSpec,
        research_state: ResearchState,
        current_params: dict[str, Any],
    ) -> AgentDecision:
        latest = research_state.latest_round
        metric_name = research_state.primary_metric_name
        metric_value = latest.primary_metric_value if latest else None
        judgement = self._latest_judgement(research_state)

        if latest is None or not metric_name or metric_value is None:
            return AgentDecision(
                action="stop",
                next_params=current_params,
                reason="No primary metric was available for the heuristic fallback agent.",
                focus_metrics=self._focus_metrics(spec),
                latest_round_judgement="inconclusive",
                avoid_repeating=research_state.rejected_change_signatures,
                confidence=0.2,
            )

        next_params = dict(current_params)
        previous = self._previous_round(research_state)
        improved = self._improved_over_previous(research_state, latest, previous)
        hypothesis = ""
        change_summary = ""
        expected_effect = ""

        if previous is None or not improved:
            adjustment = self._safe_heuristic_adjustment(
                spec,
                next_params,
                research_state.primary_metric_goal,
            )
            if adjustment is None:
                return AgentDecision(
                    action="stop",
                    next_params=spec.merge_param_values(base=current_params),
                    reason=(
                        "Heuristic fallback could not find a safe numeric tunable parameter "
                        "to adjust. Check that numeric tunable params have defaults and usable bounds."
                    ),
                    focus_metrics=[metric_name],
                    latest_round_judgement=judgement,
                    compare_to_baseline=self._comparison_text("baseline", latest, research_state.baseline_round),
                    compare_to_best=self._comparison_text("best", latest, research_state.best_so_far_round),
                    avoid_repeating=research_state.rejected_change_signatures,
                    confidence=0.2,
                )
            next_params[adjustment.name] = adjustment.after
            hypothesis = f"Adjusting {adjustment.name} may improve {metric_name}."
            change_summary = f"Set {adjustment.name} from {adjustment.before} to {adjustment.after}."
            expected_effect = f"Improve {metric_name} while keeping other configured metrics stable."
            reason = f"Heuristic fallback adjusted {adjustment.name} after observing {metric_name}={metric_value}."
        else:
            reason = f"Heuristic fallback kept parameters unchanged because {metric_name} showed acceptable progress."

        return AgentDecision(
            action="continue",
            next_params=spec.merge_param_values(next_params, base=current_params),
            reason=reason,
            focus_metrics=[metric_name],
            hypothesis=hypothesis,
            change_summary=change_summary,
            latest_round_judgement=judgement,
            compare_to_baseline=self._comparison_text("baseline", latest, research_state.baseline_round),
            compare_to_best=self._comparison_text("best", latest, research_state.best_so_far_round),
            expected_effect=expected_effect,
            avoid_repeating=research_state.rejected_change_signatures,
            confidence=0.25,
        )

    def _safe_heuristic_adjustment(
        self,
        spec: ProjectSpec,
        current_params: dict[str, Any],
        goal: str,
    ) -> Optional[ParamAdjustment]:
        for param in self._ordered_numeric_params(spec):
            if param.name in current_params:
                raw_before = current_params[param.name]
            elif param.default is not None:
                raw_before = param.default
            else:
                continue

            try:
                before = param.normalize_value(raw_before)
                candidate = self._next_numeric_candidate(param, before, goal)
                after = param.normalize_value(candidate)
            except (OverflowError, TypeError, ValueError):
                continue

            if after == before:
                continue
            return ParamAdjustment(name=param.name, before=before, after=after)
        return None

    def _ordered_numeric_params(self, spec: ProjectSpec) -> list[Any]:
        numeric_params = [param for param in spec.tunable_params if param.type in {"int", "float"}]
        preferred = self._pick_target_param(numeric_params)
        if preferred is None:
            return []
        return [preferred, *[param for param in numeric_params if param is not preferred]]

    def _next_numeric_candidate(self, param: Any, before: Any, goal: str) -> Any:
        if param.type == "int":
            step = -1 if goal == "min" else 1
            candidate = int(before) + step
            if param.min_value is not None:
                candidate = max(candidate, math.ceil(param.min_value))
            if param.max_value is not None:
                candidate = min(candidate, math.floor(param.max_value))
            return candidate

        current = float(before)
        if not math.isfinite(current):
            raise ValueError(f"non-finite value for {param.name}: {before}")
        factor = 0.8 if goal == "min" else 1.2
        candidate = current * factor
        if candidate == current:
            if goal == "min" and param.min_value is not None and current > param.min_value:
                candidate = (current + param.min_value) / 2
            elif goal == "max" and param.max_value is not None and current < param.max_value:
                candidate = (current + param.max_value) / 2
        if param.min_value is not None:
            candidate = max(candidate, param.min_value)
        if param.max_value is not None:
            candidate = min(candidate, param.max_value)
        if not math.isfinite(candidate):
            raise ValueError(f"non-finite candidate for {param.name}: {candidate}")
        return candidate

    def _pick_target_param(self, numeric_params: list[Any]) -> Any:
        for param in numeric_params:
            signature = f"{param.name} {param.flag}".lower()
            if any(token in signature for token in ("lr", "learning_rate", "learning-rate")):
                return param
        return numeric_params[0] if numeric_params else None

    def _latest_judgement(self, research_state: ResearchState) -> str:
        latest = research_state.latest_round
        baseline = research_state.baseline_round
        best = research_state.best_so_far_round
        if latest is None or latest.primary_metric_value is None:
            return "inconclusive"
        if baseline is not None and latest.round_id == baseline.round_id:
            return "baseline"
        if best is not None and latest.round_id == best.round_id:
            return "improved"
        previous = self._previous_round(research_state)
        if previous is None or previous.primary_metric_value is None:
            return "inconclusive"
        if self._is_better(
            latest.primary_metric_value,
            previous.primary_metric_value,
            research_state.primary_metric_goal,
        ):
            return "improved"
        if latest.primary_metric_value != previous.primary_metric_value:
            return "worse"
        return "inconclusive"

    def _previous_round(self, research_state: ResearchState) -> Optional[ResearchRoundState]:
        if len(research_state.recent_rounds) < 2:
            return None
        return research_state.recent_rounds[-2]

    def _improved_over_previous(
        self,
        research_state: ResearchState,
        latest: ResearchRoundState,
        previous: Optional[ResearchRoundState],
    ) -> bool:
        if previous is None or latest.primary_metric_value is None or previous.primary_metric_value is None:
            return False
        return self._is_better(
            latest.primary_metric_value,
            previous.primary_metric_value,
            research_state.primary_metric_goal,
        )

    def _comparison_text(
        self,
        label: str,
        latest: ResearchRoundState,
        reference: Optional[ResearchRoundState],
    ) -> str:
        if latest.primary_metric_value is None or reference is None or reference.primary_metric_value is None:
            return ""
        delta = latest.primary_metric_value - reference.primary_metric_value
        return f"{label}: {latest.primary_metric_value} (delta {delta:+g})"

    def _is_better(self, candidate: float, reference: float, goal: str) -> bool:
        return candidate < reference if goal == "min" else candidate > reference

    def _focus_metrics(self, spec: ProjectSpec) -> list[str]:
        if spec.metric_specs:
            return [item.name for item in spec.metric_specs]
        return ["loss", "total_loss"]

    def _extract_json(self, content: str) -> dict[str, Any]:
        content = content.strip()
        if content.startswith("{"):
            return json.loads(content)
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in completion")
        return json.loads(content[start : end + 1])
