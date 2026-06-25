from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from trainee.models import ProjectContext, ProjectSpec
from trainee.program import DEFAULT_AGENT_PROGRAM
from trainee.prompt_documents import PromptDocument
from trainee.research_state import ResearchState


class PromptEnvelope(BaseModel):
    system_prompt: str
    user_prompt: str
    static_context_json: dict[str, Any] = Field(default_factory=dict)
    dynamic_state_json: dict[str, Any] = Field(default_factory=dict)


class PromptAssembler:
    def assemble(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        research_state: ResearchState,
        current_params: dict[str, Any],
        prompt_documents: list[PromptDocument],
    ) -> PromptEnvelope:
        documents = sorted(prompt_documents, key=lambda item: (item.priority, item.path))
        rule_texts = [item.text.strip() for item in documents if item.kind == "agent_rules" and item.text.strip()]
        system_rules = "\n\n".join(rule_texts) if rule_texts else DEFAULT_AGENT_PROGRAM.strip()
        system_prompt = system_rules + "\n\n" + self._response_contract()

        static_context = {
            "cache_version": 1,
            "task": "You are a training automation agent. Decide next training params.",
            "project_context": context.model_dump(mode="json"),
            "tunable_params": [item.model_dump(mode="json") for item in spec.tunable_params],
            "metric_specs": [item.model_dump(mode="json") for item in spec.metric_specs],
            "metric_prompt": spec.metric_prompt,
            "tuning_prompt": spec.tuning_prompt,
            "prompt_documents": [
                item.model_dump(mode="json")
                for item in documents
                if item.kind != "agent_rules"
            ],
            "prompt_document_manifest": [
                {
                    "kind": item.kind,
                    "path": item.path,
                    "digest": item.digest,
                    "priority": item.priority,
                }
                for item in documents
            ],
            "output_schema": {
                "action": "continue|stop",
                "next_params": {},
                "reason": "string",
                "focus_metrics": ["string"],
                "hypothesis": "string",
                "change_summary": "string",
                "latest_round_judgement": "improved|worse|rejected|baseline|inconclusive",
                "compare_to_baseline": "string",
                "compare_to_best": "string",
                "expected_effect": "string",
                "avoid_repeating": ["string"],
                "confidence": "number between 0 and 1",
            },
        }
        dynamic_state = {
            "current_params": current_params,
            "research_state": research_state.model_dump(mode="json"),
        }
        static_text = self._compact_json(static_context)
        dynamic_text = self._compact_json(dynamic_state)
        user_prompt = (
            "<STATIC_CONTEXT>\n"
            + static_text
            + "\n</STATIC_CONTEXT>\n\n"
            + "<DYNAMIC_ROUND_STATE>\n"
            + dynamic_text
            + "\n</DYNAMIC_ROUND_STATE>"
        )
        return PromptEnvelope(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            static_context_json=static_context,
            dynamic_state_json=dynamic_state,
        )

    def _response_contract(self) -> str:
        return (
            "Runtime response contract: Respond with JSON only. "
            "Return action, next_params, reason, focus_metrics, hypothesis, change_summary, "
            "latest_round_judgement, compare_to_baseline, compare_to_best, expected_effect, "
            "avoid_repeating, and confidence. Only touch whitelisted tunable params."
        )

    def _compact_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
