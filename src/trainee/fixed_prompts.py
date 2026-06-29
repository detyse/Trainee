from __future__ import annotations

import json
from dataclasses import dataclass

from trainee.output_discovery import _system_prompt as output_discovery_system_prompt
from trainee.provider_probe import PROVIDER_TEST_SYSTEM_PROMPT, PROVIDER_TEST_USER_PROMPT
from trainee.tunable_discovery import _discovery_system_prompt as tunable_discovery_system_prompt


@dataclass(frozen=True)
class FixedPromptInfo:
    name: str
    used_for: str
    source: str
    prompt: str
    notes: str = ""


DECISION_USER_PROMPT_FIXED_PARTS = {
    "wrapper": "<STATIC_CONTEXT>...</STATIC_CONTEXT> + <DYNAMIC_ROUND_STATE>...</DYNAMIC_ROUND_STATE>",
    "static_task": "You are a training automation agent. Decide next training params.",
    "visual_observations_guidance": (
        "If research_state rounds include visual_observations, treat them as auxiliary evidence from "
        "recent plot images. Prefer numeric metrics when they conflict with visual interpretation. "
        "Use visual evidence for plateaus, instability, overfitting, qualitative regressions, or failed plots. "
        "Do not infer exact metric values from plots unless labels clearly show them."
    ),
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

TUNABLE_DISCOVERY_USER_PROMPT_FIXED_PARTS = {
    "task": "Choose candidate tuning.yaml params for a two-stage approval workflow.",
    "output_schema": {
        "candidates": [
            {
                "name": "string",
                "target_kind": "config_path|cli_flag|note",
                "target": "config path, CLI flag, or short note",
                "type": "int|float|str|bool",
                "default": "current scalar value",
                "min_value": "reasonable lower bound",
                "max_value": "reasonable upper bound",
                "choices": ["optional string choices"],
                "applicability": "auto_applyable|needs_review|unsupported",
                "risk": "low|medium|high|unknown",
                "reason": "why this parameter is safe/useful or why review is needed",
                "evidence": ["short supporting signals from config/context"],
                "confidence": "0..1",
            }
        ]
    },
}

OUTPUT_DISCOVERY_USER_PROMPT_FIXED_PARTS = {
    "goal": "Find the config key that controls where the run writes logs, checkpoints, renders, params, or result artifacts.",
    "rules": [
        "Select only an existing key from scalar_config_leaves.",
        "Prefer output.*, output_dir, save_dir, result_dir, log_dir, or checkpoint_dir.",
        "Reject data/input/image/annotation/mask/camera/dataset paths.",
        "If no single confident output directory key exists, return selected=null.",
        "Trainee will assign the runtime output value; only choose the config key.",
    ],
    "output_schema": {
        "selected": {
            "config_path": "existing key from scalar_config_leaves",
            "current_value": "current value",
            "confidence": "0..1",
            "reason": "short reason",
            "evidence": ["short evidence"],
        },
        "candidates": [
            {
                "config_path": "existing key",
                "current_value": "current value",
                "confidence": "0..1",
                "reason": "short reason",
                "evidence": ["short evidence"],
            }
        ],
        "warnings": ["optional warning"],
    },
}

LLM_TEST_SYSTEM_PROMPT = "You are a concise API test assistant. Answer the user's prompt directly."

VISUAL_ANALYSIS_SYSTEM_PROMPT = (
    "You are a conservative visual analysis assistant for machine learning training diagnostics. "
    "Return compact JSON only. Do not recommend parameter changes directly; summarize visual evidence."
)

VISUAL_ANALYSIS_USER_PROMPT_TEMPLATE = (
    "Analyze this image from the latest completed training round.\n\n"
    "Project visual prompt: {project_visual_prompt}\n"
    "Image path: {image_path}\n\n"
    "Interpret it as an ordinary ML training/evaluation plot unless visible labels indicate otherwise.\n"
    "Use visible text first: title, axis labels, legends, filenames, and directory names.\n"
    "Then use curve shape and visual patterns.\n\n"
    "Common meanings:\n"
    "- decreasing loss/error usually suggests optimization progress\n"
    "- flat validation metric may indicate a plateau\n"
    "- training improves while validation worsens may indicate overfitting\n"
    "- spikes or oscillation may indicate instability or overly aggressive settings\n"
    "- NaN-like, blank, or collapsed plots indicate failed or invalid runs\n"
    "- qualitative render/result plots should be judged conservatively\n\n"
    "Do not choose next parameters. If the plot is unclear, say it is unclear and lower confidence.\n"
    "Return JSON only with keys: likely_meaning, visible_signals, concerns, "
    "decision_relevant_observations, confidence."
)


def fixed_prompt_inventory(system_prompt: str) -> list[FixedPromptInfo]:
    return [
        FixedPromptInfo(
            name="Training decision system prompt",
            used_for="Every LLM decision that chooses continue/stop and next tunable parameter values.",
            source="Global config system_prompt; default from src/trainee/defaults/system_prompt.txt",
            prompt=system_prompt,
            notes="Global and editable. It does not come from project.yaml or tuning.yaml.",
        ),
        FixedPromptInfo(
            name="Training decision user prompt fixed frame",
            used_for="The fixed wrapper, task text, visual guidance, and response schema around project/runtime JSON.",
            source="src/trainee/prompt_assembler.py",
            prompt=json.dumps(DECISION_USER_PROMPT_FIXED_PARTS, ensure_ascii=False, indent=2),
            notes="Project context, tunables, metrics, history, and prompt documents are inserted at runtime.",
        ),
        FixedPromptInfo(
            name="Tunable discovery system prompt",
            used_for="LLM-assisted Suggest Tunables.",
            source="src/trainee/tunable_discovery.py",
            prompt=tunable_discovery_system_prompt(),
        ),
        FixedPromptInfo(
            name="Tunable discovery user prompt fixed parts",
            used_for="The fixed task and output schema for LLM-assisted Suggest Tunables.",
            source="src/trainee/tunable_discovery.py",
            prompt=json.dumps(TUNABLE_DISCOVERY_USER_PROMPT_FIXED_PARTS, ensure_ascii=False, indent=2),
            notes="Project context, baseline config leaves, current tunables, and prompt documents are inserted at runtime.",
        ),
        FixedPromptInfo(
            name="Output discovery system prompt",
            used_for="LLM-assisted detection of the output directory config key.",
            source="src/trainee/output_discovery.py",
            prompt=output_discovery_system_prompt(),
        ),
        FixedPromptInfo(
            name="Output discovery user prompt fixed parts",
            used_for="The fixed goal, rules, and output schema for output config discovery.",
            source="src/trainee/output_discovery.py",
            prompt=json.dumps(OUTPUT_DISCOVERY_USER_PROMPT_FIXED_PARTS, ensure_ascii=False, indent=2),
            notes="Project context, baseline config leaves, and prompt documents are inserted at runtime.",
        ),
        FixedPromptInfo(
            name="Visual analysis system prompt",
            used_for="Image analysis for training diagnostic plots.",
            source="src/trainee/decision.py",
            prompt=VISUAL_ANALYSIS_SYSTEM_PROMPT,
        ),
        FixedPromptInfo(
            name="Visual analysis user prompt template",
            used_for="Per-image visual analysis request.",
            source="src/trainee/visuals.py",
            prompt=VISUAL_ANALYSIS_USER_PROMPT_TEMPLATE,
            notes="The project-specific visual prompt and image path fill the placeholders.",
        ),
        FixedPromptInfo(
            name="Provider live test system prompt",
            used_for="Provider Settings live API probe.",
            source="src/trainee/provider_probe.py",
            prompt=PROVIDER_TEST_SYSTEM_PROMPT,
        ),
        FixedPromptInfo(
            name="Provider live test user prompt",
            used_for="Provider Settings live API probe.",
            source="src/trainee/provider_probe.py",
            prompt=PROVIDER_TEST_USER_PROMPT,
        ),
        FixedPromptInfo(
            name="LLM API test system prompt",
            used_for="The manual /llm-test page.",
            source="src/trainee/decision.py",
            prompt=LLM_TEST_SYSTEM_PROMPT,
            notes="The user message on that page is typed by the user, not project-derived.",
        ),
    ]
