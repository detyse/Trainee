from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from trainee.models import AgentDecision, AgentTrace, ProjectContext, ProjectSpec, PromptPreview
from trainee.prompt_assembler import PromptAssembler, PromptEnvelope
from trainee.prompt_documents import PromptDocument
from trainee.research_state import ResearchRoundState, ResearchState
from trainee.settings import Settings


@dataclass
class DecisionResult:
    decision: AgentDecision
    prompt_preview: Optional[PromptPreview] = None
    agent_trace: Optional[AgentTrace] = None


@dataclass
class ProviderCompletion:
    content: str
    raw_response_body: str
    http_status: int
    request_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str,
        http_status: Optional[int] = None,
        request_id: Optional[str] = None,
        raw_response_body: Optional[str] = None,
        error_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.http_status = http_status
        self.request_id = request_id
        self.raw_response_body = raw_response_body
        self.error_body = error_body


class DecisionEngine:
    def __init__(self, settings: Settings, prompt_assembler: Optional[PromptAssembler] = None):
        self.settings = settings
        self.prompt_assembler = prompt_assembler or PromptAssembler()

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
        )
        prompt_preview = self._build_prompt_preview(envelope, status="not_sent")
        agent_trace: Optional[AgentTrace] = None

        if self._provider_is_configured():
            prompt_preview = self._build_prompt_preview(envelope, status="sent")
            try:
                completion = await self._provider_complete(envelope.system_prompt, envelope.user_prompt)
            except ProviderCallError as exc:
                agent_trace = self._new_trace(
                    status=exc.status,
                    http_status=exc.http_status,
                    request_id=exc.request_id,
                    raw_response_body=exc.raw_response_body,
                    error_body=exc.error_body,
                    provider_error=str(exc),
                    fallback_reason=str(exc),
                )
                prompt_preview = prompt_preview.model_copy(update={"status": exc.status})
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                agent_trace = self._new_trace(
                    status="response_failed",
                    provider_error=message,
                    fallback_reason=message,
                )
                prompt_preview = prompt_preview.model_copy(update={"status": "response_failed"})
            else:
                agent_trace = self._new_trace(
                    status="success",
                    http_status=completion.http_status,
                    request_id=completion.request_id,
                    raw_response_body=completion.raw_response_body,
                    raw_output=completion.content,
                    usage=completion.usage,
                    finish_reason=completion.finish_reason,
                )
                try:
                    candidate = self._extract_json(completion.content)
                except (ValueError, json.JSONDecodeError) as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if agent_trace is not None:
                        agent_trace.status = "parse_failed"
                        agent_trace.parse_error = message
                        agent_trace.fallback_reason = message
                    prompt_preview = prompt_preview.model_copy(update={"status": "parse_failed"})
                else:
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
                        prompt_preview = prompt_preview.model_copy(update={"status": "validation_failed"})
                    else:
                        try:
                            decision.next_params = spec.merge_param_values(decision.next_params, base=current_params)
                        except ValueError as exc:
                            message = f"tunable parameter validation failed: {exc}"
                            if agent_trace is not None:
                                agent_trace.status = "validation_failed"
                                agent_trace.validation_error = message
                                agent_trace.fallback_reason = message
                            prompt_preview = prompt_preview.model_copy(update={"status": "validation_failed"})
                        else:
                            return DecisionResult(
                                decision=decision,
                                prompt_preview=prompt_preview,
                                agent_trace=agent_trace,
                            )
        else:
            reason = self._provider_unavailable_reason()
            agent_trace = self._new_trace(status="not_called", fallback_reason=reason)

        return DecisionResult(
            decision=self._heuristic_decision(spec, research_state, current_params),
            prompt_preview=prompt_preview,
            agent_trace=agent_trace,
        )

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
        )
        return self._build_prompt_preview(envelope, status=status)

    async def probe(self, prompt: str, image: Optional[dict[str, str]] = None) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")

        if self.settings.llm_provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            completion = await self._openai_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "openai",
                "model": self.settings.openai_model,
                "has_image": image is not None,
                "content": completion.content,
            }
        if self.settings.llm_provider == "moonshot":
            if not self.settings.moonshot_api_key:
                raise ValueError("MOONSHOT_API_KEY is not configured")
            completion = await self._moonshot_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "moonshot",
                "model": self.settings.moonshot_model,
                "has_image": image is not None,
                "content": completion.content,
            }
        if self.settings.llm_provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not configured")
            completion = await self._anthropic_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "anthropic",
                "model": self.settings.anthropic_model,
                "has_image": image is not None,
                "content": completion.content,
            }
        raise ValueError("LLM provider is disabled")

    def _build_prompt_preview(self, envelope: PromptEnvelope, status: str) -> PromptPreview:
        provider = self.settings.llm_provider
        model = "none"
        payload: dict[str, Any] = {
            "system": envelope.system_prompt,
            "user": envelope.user_prompt,
        }
        if provider == "openai":
            model = self.settings.openai_model
            payload = self._openai_payload(envelope.system_prompt, envelope.user_prompt)
        elif provider == "moonshot":
            model = self.settings.moonshot_model
            payload = self._moonshot_payload(envelope.system_prompt, envelope.user_prompt)
        elif provider == "anthropic":
            model = self.settings.anthropic_model
            payload = self._anthropic_payload(envelope.system_prompt, envelope.user_prompt)
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

    def _provider_is_configured(self) -> bool:
        if self.settings.llm_provider == "openai":
            return bool(self.settings.openai_api_key)
        if self.settings.llm_provider == "moonshot":
            return bool(self.settings.moonshot_api_key)
        if self.settings.llm_provider == "anthropic":
            return bool(self.settings.anthropic_api_key)
        return False

    def _provider_unavailable_reason(self) -> str:
        provider = self.settings.llm_provider
        if provider == "none":
            return "LLM provider is disabled."
        return f"{provider.upper()} API key is not configured."

    def _provider_model(self, provider: str) -> Optional[str]:
        if provider == "openai":
            return self.settings.openai_model
        if provider == "moonshot":
            return self.settings.moonshot_model
        if provider == "anthropic":
            return self.settings.anthropic_model
        return None

    def _new_trace(self, *, status: str, **updates: Any) -> Optional[AgentTrace]:
        if not self.settings.agent_debug_enabled:
            return None
        provider = self.settings.llm_provider
        return AgentTrace(
            provider=provider,
            model=self._provider_model(provider),
            status=status,
            **updates,
        )

    async def _provider_complete(self, system_prompt: str, user_prompt: str) -> ProviderCompletion:
        if self.settings.llm_provider == "openai":
            return await self._openai_complete(system_prompt, user_prompt)
        if self.settings.llm_provider == "moonshot":
            return await self._moonshot_complete(system_prompt, user_prompt)
        if self.settings.llm_provider == "anthropic":
            return await self._anthropic_complete(system_prompt, user_prompt)
        raise ProviderCallError("LLM provider is disabled.", status="not_called")

    async def _openai_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderCompletion:
        payload = self._openai_payload(system_prompt, user_prompt, image=image)
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        body, raw_body, http_status, request_id = await self._post_provider_json(url, payload, headers)
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content must be a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise self._response_shape_error(
                "OpenAI response did not contain choices[0].message.content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None,
        )

    async def _moonshot_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderCompletion:
        payload = self._moonshot_payload(system_prompt, user_prompt, image=image)
        url = self.settings.moonshot_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.moonshot_api_key}"}
        body, raw_body, http_status, request_id = await self._post_provider_json(url, payload, headers)
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content must be a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise self._response_shape_error(
                "Moonshot response did not contain choices[0].message.content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None,
        )

    async def _anthropic_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderCompletion:
        payload = self._anthropic_payload(system_prompt, user_prompt, image=image)
        url = self.settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": self.settings.anthropic_version,
            "content-type": "application/json",
        }
        body, raw_body, http_status, request_id = await self._post_provider_json(url, payload, headers)
        try:
            text_parts = [
                block.get("text", "")
                for block in body["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(part for part in text_parts if part)
            if not content:
                raise ValueError("no text content blocks")
        except (KeyError, TypeError, ValueError) as exc:
            raise self._response_shape_error(
                "Anthropic response did not contain text content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(body["stop_reason"]) if body.get("stop_reason") is not None else None,
        )

    async def _post_provider_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], str, int, Optional[str]]:
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            raise ProviderCallError(
                f"{type(exc).__name__}: {exc}",
                status="request_failed",
            ) from exc

        raw_body = response.text
        request_id = self._request_id(response)
        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderCallError(
                f"Provider returned HTTP {response.status_code}.",
                status="request_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
                error_body=raw_body,
            )
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderCallError(
                f"Provider response body was not valid JSON: {exc}",
                status="response_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
            ) from exc
        if not isinstance(body, dict):
            raise ProviderCallError(
                "Provider response body must be a JSON object.",
                status="response_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
            )
        return body, raw_body, response.status_code, request_id

    def _response_shape_error(
        self,
        message: str,
        exc: Exception,
        raw_body: str,
        http_status: int,
        request_id: Optional[str],
    ) -> ProviderCallError:
        return ProviderCallError(
            f"{message}: {exc}",
            status="response_failed",
            http_status=http_status,
            request_id=request_id,
            raw_response_body=raw_body,
        )

    def _request_id(self, response: httpx.Response) -> Optional[str]:
        for header in ("x-request-id", "request-id", "openai-request-id"):
            value = response.headers.get(header)
            if value:
                return value
        return None

    def _openai_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._openai_user_content(user_prompt, image)},
            ],
            "temperature": 0.2,
        }

    def _moonshot_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.moonshot_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._openai_user_content(user_prompt, image)},
            ],
            "temperature": 0.2,
        }

    def _anthropic_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.anthropic_max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": self._anthropic_user_content(user_prompt, image),
                }
            ],
            "temperature": 0.2,
        }

    def _openai_user_content(self, prompt: str, image: Optional[dict[str, str]]) -> Any:
        if image is None:
            return prompt
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image['media_type']};base64,{image['data']}"},
            },
        ]

    def _anthropic_user_content(self, prompt: str, image: Optional[dict[str, str]]) -> Any:
        if image is None:
            return prompt
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["media_type"],
                    "data": image["data"],
                },
            },
        ]

    def _probe_system_prompt(self) -> str:
        return "You are a concise API test assistant. Answer the user's prompt directly."

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
        numeric_params = [param for param in spec.tunable_params if param.type in {"int", "float"}]
        target_param = self._pick_target_param(numeric_params)
        previous = self._previous_round(research_state)
        improved = self._improved_over_previous(research_state, latest, previous)
        hypothesis = ""
        change_summary = ""
        expected_effect = ""

        if target_param is not None and (previous is None or not improved):
            before = target_param.normalize_value(next_params.get(target_param.name, target_param.default))
            if target_param.type == "int":
                step = -1 if research_state.primary_metric_goal == "min" else 1
                candidate = int(before) + step
            else:
                factor = 0.8 if research_state.primary_metric_goal == "min" else 1.2
                candidate = float(before) * factor
            after = target_param.normalize_value(candidate)
            next_params[target_param.name] = after
            hypothesis = f"Adjusting {target_param.name} may improve {metric_name}."
            change_summary = f"Set {target_param.name} from {before} to {after}."
            expected_effect = f"Improve {metric_name} while keeping other configured metrics stable."
            reason = f"Heuristic fallback adjusted {target_param.name} after observing {metric_name}={metric_value}."
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
