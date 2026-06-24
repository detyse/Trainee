from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from trainee.models import AgentDecision, ProjectContext, ProjectSpec, PromptPreview, RoundRecord
from trainee.settings import Settings


@dataclass
class DecisionResult:
    decision: AgentDecision
    prompt_preview: Optional[PromptPreview] = None


class DecisionEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def decide(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> AgentDecision:
        result = await self.decide_with_prompt(spec, context, history, current_params)
        return result.decision

    async def decide_with_prompt(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> DecisionResult:
        if not spec.tunable_params:
            return DecisionResult(
                decision=AgentDecision(
                    action="stop",
                    next_params=current_params,
                    reason="No tunable_params are configured, so the loop stops after collecting the latest result.",
                    focus_metrics=self._focus_metrics(spec),
                )
            )

        decision, prompt_preview = await self._provider_decision(spec, context, history, current_params)
        if decision is not None:
            try:
                normalized = spec.merge_param_values(decision.next_params, base=current_params)
            except ValueError:
                pass
            else:
                decision.next_params = normalized
                return DecisionResult(decision=decision, prompt_preview=prompt_preview)

        if prompt_preview is None:
            prompt_preview = self.build_prompt_preview(spec, context, history, current_params, status="not_sent")

        return DecisionResult(
            decision=self._heuristic_decision(spec, history, current_params),
            prompt_preview=prompt_preview,
        )

    async def _provider_decision(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> Tuple[Optional[AgentDecision], Optional[PromptPreview]]:
        if self.settings.llm_provider == "openai" and self.settings.openai_api_key:
            return await self._openai_decision(spec, context, history, current_params)
        if self.settings.llm_provider == "moonshot" and self.settings.moonshot_api_key:
            return await self._moonshot_decision(spec, context, history, current_params)
        if self.settings.llm_provider == "anthropic" and self.settings.anthropic_api_key:
            return await self._anthropic_decision(spec, context, history, current_params)
        return None, None

    async def _openai_decision(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> Tuple[Optional[AgentDecision], Optional[PromptPreview]]:
        prompt = self._build_prompt(spec, context, history, current_params)
        prompt_preview = self.build_prompt_preview(spec, context, history, current_params, status="sent")
        try:
            content = await self._openai_complete(self._system_prompt(), prompt)
            candidate = self._extract_json(content)
            return AgentDecision.model_validate(candidate), prompt_preview
        except Exception:
            return None, prompt_preview.model_copy(update={"status": "provider_failed"})

    async def _moonshot_decision(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> Tuple[Optional[AgentDecision], Optional[PromptPreview]]:
        prompt = self._build_prompt(spec, context, history, current_params)
        prompt_preview = self.build_prompt_preview(spec, context, history, current_params, status="sent")
        try:
            content = await self._moonshot_complete(self._system_prompt(), prompt)
            candidate = self._extract_json(content)
            return AgentDecision.model_validate(candidate), prompt_preview
        except Exception:
            return None, prompt_preview.model_copy(update={"status": "provider_failed"})

    async def _anthropic_decision(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> Tuple[Optional[AgentDecision], Optional[PromptPreview]]:
        prompt = self._build_prompt(spec, context, history, current_params)
        prompt_preview = self.build_prompt_preview(spec, context, history, current_params, status="sent")
        try:
            content = await self._anthropic_complete(self._system_prompt(), prompt)
            candidate = self._extract_json(content)
            return AgentDecision.model_validate(candidate), prompt_preview
        except Exception:
            return None, prompt_preview.model_copy(update={"status": "provider_failed"})

    async def probe(self, prompt: str, image: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is required")

        if self.settings.llm_provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured")
            content = await self._openai_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "openai",
                "model": self.settings.openai_model,
                "has_image": image is not None,
                "content": content,
            }

        if self.settings.llm_provider == "moonshot":
            if not self.settings.moonshot_api_key:
                raise ValueError("MOONSHOT_API_KEY is not configured")
            content = await self._moonshot_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "moonshot",
                "model": self.settings.moonshot_model,
                "has_image": image is not None,
                "content": content,
            }

        if self.settings.llm_provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not configured")
            content = await self._anthropic_complete(self._probe_system_prompt(), prompt, image=image)
            return {
                "provider": "anthropic",
                "model": self.settings.anthropic_model,
                "has_image": image is not None,
                "content": content,
            }

        raise ValueError("LLM provider is disabled")

    async def _openai_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Dict[str, str]] = None,
    ) -> str:
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._openai_user_content(user_prompt, image)},
            ],
            "temperature": 0.2,
        }
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _moonshot_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Dict[str, str]] = None,
    ) -> str:
        payload = self._moonshot_payload(system_prompt, user_prompt, image=image)
        url = self.settings.moonshot_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.moonshot_api_key}"}
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            print("moonshot usage:", body.get("usage"))
            return body["choices"][0]["message"]["content"]

    async def _anthropic_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Dict[str, str]] = None,
    ) -> str:
        payload = {
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
        url = self.settings.anthropic_base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": self.settings.anthropic_version,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
            text_parts = [
                block.get("text", "")
                for block in body.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part)

    def _openai_user_content(self, prompt: str, image: Optional[Dict[str, str]]) -> Any:
        if image is None:
            return prompt
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image['media_type']};base64,{image['data']}",
                },
            },
        ]

    def _anthropic_user_content(self, prompt: str, image: Optional[Dict[str, str]]) -> Any:
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

    def _system_prompt(self) -> str:
        return (
            "You are a training automation agent. Respond with JSON only. "
            "Schema: {\"action\":\"continue|stop\",\"next_params\":{...},"
            "\"reason\":\"...\",\"focus_metrics\":[\"...\"]}. "
            "Only touch whitelisted tunable params."
        )

    def _probe_system_prompt(self) -> str:
        return "You are a concise API test assistant. Answer the user's prompt directly."

    def _heuristic_decision(
        self,
        spec: ProjectSpec,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> AgentDecision:
        latest = history[-1]
        primary_metric = self._primary_metric(spec, latest.metrics)
        next_params = dict(current_params)
        numeric_params = [param for param in spec.tunable_params if param.type in {"int", "float"}]

        if primary_metric is None:
            return AgentDecision(
                action="stop",
                next_params=current_params,
                reason="No primary metric was available for the heuristic fallback agent.",
                focus_metrics=self._focus_metrics(spec),
            )

        metric_name = primary_metric.name
        current_value = float(latest.metrics[metric_name])
        previous_value = None
        if len(history) > 1 and metric_name in history[-2].metrics:
            previous_value = float(history[-2].metrics[metric_name])

        improvement = True
        if previous_value is not None:
            if primary_metric.goal == "min":
                improvement = current_value <= previous_value
            else:
                improvement = current_value >= previous_value

        target_param = self._pick_target_param(spec, numeric_params)
        if target_param is not None and (previous_value is None or not improvement):
            baseline = next_params.get(target_param.name, target_param.default)
            baseline = target_param.normalize_value(baseline)
            if target_param.type == "int":
                step = -1 if primary_metric.goal == "min" else 1
                candidate = int(baseline) + step
            else:
                factor = 0.8 if primary_metric.goal == "min" else 1.2
                candidate = float(baseline) * factor
            next_params[target_param.name] = target_param.normalize_value(candidate)
            reason = (
                f"Heuristic fallback adjusted {target_param.name} after observing {metric_name}={current_value}."
            )
        else:
            reason = (
                f"Heuristic fallback kept parameters unchanged because {metric_name} showed acceptable progress."
            )

        return AgentDecision(
            action="continue",
            next_params=spec.merge_param_values(next_params, base=current_params),
            reason=reason,
            focus_metrics=[metric_name],
        )

    def _pick_target_param(self, spec: ProjectSpec, numeric_params: List[Any]) -> Any:
        for param in numeric_params:
            signature = f"{param.name} {param.flag}".lower()
            if any(token in signature for token in ("lr", "learning_rate", "learning-rate")):
                return param
        return numeric_params[0] if numeric_params else None

    def _primary_metric(self, spec: ProjectSpec, metrics: Dict[str, Any]):
        if spec.metric_specs:
            for item in spec.metric_specs:
                if item.name in metrics:
                    return item
        for fallback_name in ("total_loss", "loss"):
            if fallback_name in metrics:
                goal = "min"
                return spec.metric_index().get(
                    fallback_name,
                    type("FallbackMetric", (), {"name": fallback_name, "goal": goal})(),
                )
        return None

    def _focus_metrics(self, spec: ProjectSpec) -> List[str]:
        if spec.metric_specs:
            return [item.name for item in spec.metric_specs]
        return ["loss", "total_loss"]

    def build_prompt_preview(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
        status: str = "preview",
    ) -> PromptPreview:
        system_prompt = self._system_prompt()
        user_prompt = self._build_prompt(spec, context, history, current_params)
        provider = self.settings.llm_provider
        model = "none"
        payload: Dict[str, Any] = {"system": system_prompt, "user": user_prompt}
        if provider == "openai":
            model = self.settings.openai_model
            payload = self._openai_payload(system_prompt, user_prompt)
        elif provider == "moonshot":
            model = self.settings.moonshot_model
            payload = self._moonshot_payload(system_prompt, user_prompt)
        elif provider == "anthropic":
            model = self.settings.anthropic_model
            payload = self._anthropic_payload(system_prompt, user_prompt)
        return PromptPreview(
            provider=provider,
            model=model,
            status=status,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            payload=payload,
        )

    def _openai_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
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
        image: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
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
        image: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
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

    def _build_prompt(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        history: List[RoundRecord],
        current_params: Dict[str, Any],
    ) -> str:
        static_payload = {
            "cache_version": 1,
            "task": "You are a training automation agent. Decide next training params.",
            "project_context": context.model_dump(mode="json"),
            "tunable_params": [item.model_dump(mode="json") for item in spec.tunable_params],
            "metric_specs": [item.model_dump(mode="json") for item in spec.metric_specs],
            "metric_prompt": spec.metric_prompt,
            "tuning_prompt": spec.tuning_prompt,
            "output_schema": {
                "action": "continue|stop",
                "next_params": {},
                "reason": "string",
                "focus_metrics": ["string"],
            },
        }

        recent_history = []
        for item in history[-5:]:
            recent_history.append(
                {
                    "round_index": item.round_index,
                    "status": item.status,
                    "param_values": item.param_values,
                    "metrics": item.metrics,
                    "exit_code": item.exit_code,
                }
            )

        dynamic_payload = {
            "current_params": current_params,
            "recent_history": recent_history,
        }

        static_text = json.dumps(
            static_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        dynamic_text = json.dumps(
            dynamic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return (
            "<STATIC_CONTEXT>\n"
            + static_text
            + "\n</STATIC_CONTEXT>\n\n"
            + "<DYNAMIC_ROUND_STATE>\n"
            + dynamic_text
            + "\n</DYNAMIC_ROUND_STATE>"
        )

    def _extract_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("{"):
            return json.loads(content)
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in completion")
        return json.loads(content[start : end + 1])
    
    # add decision prompt here? 
    def _decision_policy(self, ):
        
        return 
    

    
