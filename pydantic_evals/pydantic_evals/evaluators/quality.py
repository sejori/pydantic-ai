"""Curated presets of LLM-backed quality evaluators.

This module provides thin, named wrappers over [`LLMJudge`][pydantic_evals.evaluators.LLMJudge]
and the [`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers, with rubrics aligned to
widely-used evaluation frameworks such as [Ragas](https://github.com/explodinggradients/ragas),
G-Eval (Liu et al., 2023) and GEMBA (Kocmi & Federmann, 2023).

These evaluators are intentionally lightweight — you can always reach for `LLMJudge` directly with
a bespoke rubric. Their purpose is to provide recognisable names and sensible defaults for the
common quality dimensions users expect to see.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings

from .common import OutputConfig
from .context import EvaluatorContext
from .evaluator import EvaluationReason, EvaluationScalar, Evaluator, EvaluatorOutput
from .llm_as_a_judge import (
    GradingOutput,
    judge_g_eval,
    judge_gemba_da,
    judge_gemba_sqm,
    judge_input_output,
    judge_input_output_expected,
)

__all__ = (
    'AnswerRelevance',
    'ContextPrecision',
    'ContextRecall',
    'ContextSource',
    'Faithfulness',
    'GEval',
    'GembaScore',
    'Hallucination',
    'ValueSource',
)


ValueSource = str | Callable[[EvaluatorContext[object, object, object]], str]
"""A value that is either a fixed string or a callable that derives the string from the context."""

ContextSource = ValueSource
"""Alias for [`ValueSource`][pydantic_evals.evaluators.quality.ValueSource], kept for readability in evaluators that take a retrieved context."""


def _resolve(
    value: ValueSource,
    ctx: EvaluatorContext[object, object, object],
) -> str:
    """Resolve a [`ValueSource`][pydantic_evals.evaluators.quality.ValueSource] to a string."""
    if callable(value):
        return value(ctx)
    return value


def _default_score_config() -> OutputConfig:
    return OutputConfig(include_reason=True)


def _default_assertion_config() -> OutputConfig:
    return OutputConfig(include_reason=True)


def _write_result(
    output: dict[str, EvaluationScalar | EvaluationReason],
    value: EvaluationScalar,
    reason: str | None,
    config: OutputConfig,
    default_name: str,
) -> None:
    name = config.get('evaluation_name') or default_name
    if config.get('include_reason') and reason is not None:
        output[name] = EvaluationReason(value=value, reason=reason)
    else:
        output[name] = value


def _emit(
    evaluator: Evaluator[object, object, object],
    grading_output: GradingOutput,
    score_config: OutputConfig | Literal[False],
    assertion_config: OutputConfig | Literal[False],
) -> EvaluatorOutput:
    """Build an [`EvaluatorOutput`][pydantic_evals.evaluators.EvaluatorOutput] from a [`GradingOutput`][pydantic_evals.evaluators.llm_as_a_judge.GradingOutput].

    Mirrors the score/assertion naming logic used by [`LLMJudge`][pydantic_evals.evaluators.LLMJudge]
    so quality-pack evaluators produce the same shape of reports.
    """
    output: dict[str, EvaluationScalar | EvaluationReason] = {}
    include_both = score_config is not False and assertion_config is not False
    evaluation_name = evaluator.get_default_evaluation_name()

    if score_config is not False:
        default_name = f'{evaluation_name}_score' if include_both else evaluation_name
        _write_result(output, grading_output.score, grading_output.reason, score_config, default_name)

    if assertion_config is not False:
        default_name = f'{evaluation_name}_pass' if include_both else evaluation_name
        _write_result(output, grading_output.pass_, grading_output.reason, assertion_config, default_name)

    return output


_FAITHFULNESS_RUBRIC = dedent(
    """\
    The Input is a JSON object with two fields: `question` (the user question) and `context`
    (the retrieved context that the Output is supposed to rely on).

    Every factual claim in the Output must be directly supported by `context`. Pass only if all
    claims are entailed by the context. Unsupported claims, contradictions, or fabrications
    constitute failure. Ignore information that is correct in the real world but not present in
    the provided context.

    The score should be the fraction of factual claims that are supported by the context
    (0.0 = none are supported, 1.0 = all are supported).
    """
)


@dataclass(repr=False)
class Faithfulness(Evaluator[object, object, object]):
    """Ragas-style faithfulness: every factual claim in the output is supported by the provided context.

    Pass when all claims are grounded in `context`; score reflects the fraction of claims that are
    supported (higher is better).
    """

    context: ContextSource
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        resolved_context = _resolve(self.context, ctx)
        judge_inputs = {'question': ctx.inputs, 'context': resolved_context}
        grading_output = await judge_input_output(
            judge_inputs, ctx.output, _FAITHFULNESS_RUBRIC, self.model, self.model_settings
        )
        return _emit(self, grading_output, self.score, self.assertion)


_ANSWER_RELEVANCE_RUBRIC = dedent(
    """\
    The Output must directly answer the question in the Input.

    A strong answer:
    - Addresses the question without redundancy or padding,
    - Stays on topic (no unrelated tangents),
    - Gives the level of detail the question calls for (terse when asked terse, thorough when asked thorough).

    The score should reflect how directly and completely the Output answers the question
    (0.0 = unrelated, 1.0 = a direct, on-point answer).
    """
)


@dataclass(repr=False)
class AnswerRelevance(Evaluator[object, object, object]):
    """Ragas-style answer relevance: the output directly addresses the input question.

    By default the question is pulled from `ctx.inputs`. Pass `question` as a callable to extract
    it from a richer input structure.
    """

    question: ValueSource | None = None
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        judge_input = _resolve(self.question, ctx) if self.question is not None else ctx.inputs
        grading_output = await judge_input_output(
            judge_input, ctx.output, _ANSWER_RELEVANCE_RUBRIC, self.model, self.model_settings
        )
        return _emit(self, grading_output, self.score, self.assertion)


_CONTEXT_PRECISION_RUBRIC = dedent(
    """\
    The Input is a JSON object with fields `question` (the user question) and `context`
    (the retrieved context being evaluated). The Output echoes the same `context` string —
    it is the thing being judged, not a generated answer.

    Identify which portions of `context` are actually relevant to answering the `question`,
    and which are irrelevant filler. A high-precision retrieval surfaces only relevant passages.

    The score should be the fraction of the context that is relevant to the question
    (0.0 = none is relevant, 1.0 = all of it is relevant). Pass when precision is high enough
    that irrelevant content would not materially distract a downstream model.
    """
)


@dataclass(repr=False)
class ContextPrecision(Evaluator[object, object, object]):
    """Ragas-style context precision: how much of the retrieved context is relevant to the question.

    The evaluator treats the retrieved context as the thing being judged. By default the question
    is drawn from `ctx.inputs`.
    """

    context: ContextSource
    question: ValueSource | None = None
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        resolved_context = _resolve(self.context, ctx)
        resolved_question = _resolve(self.question, ctx) if self.question is not None else ctx.inputs
        judge_inputs = {'question': resolved_question, 'context': resolved_context}
        grading_output = await judge_input_output(
            judge_inputs, resolved_context, _CONTEXT_PRECISION_RUBRIC, self.model, self.model_settings
        )
        return _emit(self, grading_output, self.score, self.assertion)


_CONTEXT_RECALL_RUBRIC = dedent(
    """\
    The Input is a JSON object with fields `question` (the user question) and `context`
    (the retrieved context). The ExpectedOutput contains the ground-truth answer.

    Determine whether `context` contains enough information to produce the ground-truth answer.
    A high-recall retrieval leaves no gaps: every claim in the ground-truth answer should be
    supported by `context`.

    The score should be the fraction of the ground-truth answer that is supported by `context`
    (0.0 = none of it is covered, 1.0 = all of it is covered). Pass when `context` is sufficient
    to fully produce the ground-truth answer.
    """
)


@dataclass(repr=False)
class ContextRecall(Evaluator[object, object, object]):
    """Ragas-style context recall: whether the retrieved context is sufficient to produce the ground-truth answer.

    By default the ground truth comes from `ctx.expected_output`; pass `ground_truth` as a callable
    to extract it from a richer structure.
    """

    context: ContextSource
    ground_truth: ValueSource | None = None
    question: ValueSource | None = None
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        resolved_context = _resolve(self.context, ctx)
        resolved_question = _resolve(self.question, ctx) if self.question is not None else ctx.inputs
        if self.ground_truth is not None:
            resolved_ground_truth = _resolve(self.ground_truth, ctx)
        elif ctx.expected_output is None:
            return {}
        else:
            resolved_ground_truth = ctx.expected_output
        judge_inputs = {'question': resolved_question, 'context': resolved_context}
        grading_output = await judge_input_output_expected(
            judge_inputs,
            ctx.output,
            resolved_ground_truth,
            _CONTEXT_RECALL_RUBRIC,
            self.model,
            self.model_settings,
        )
        return _emit(self, grading_output, self.score, self.assertion)


_HALLUCINATION_RUBRIC = dedent(
    """\
    The Input is a JSON object with fields `question` (the user question) and `context`
    (the retrieved context that the Output is supposed to rely on).

    Evaluate whether the Output contains factual claims that are NOT supported by `context`.
    Unsupported or fabricated claims count as hallucinations even if they happen to be true in
    the real world.

    Score the fraction of factual claims that ARE supported by the context
    (0.0 = fully hallucinated, 1.0 = fully grounded). Set pass=true ONLY if one or more
    hallucinations are present; otherwise set pass=false. In other words, `pass` is a detector
    that fires when hallucination is present — the assertion is intentionally inverted relative
    to the groundedness score.
    """
)


@dataclass(repr=False)
class Hallucination(Evaluator[object, object, object]):
    """Detect unsupported (hallucinated) claims in the output relative to a provided context.

    This is the sibling of [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] framed as a
    detector:

    - `score` is a groundedness score (higher = more grounded, lower = more hallucination).
    - The assertion fires (`pass=True`) when hallucination is detected — i.e. when the output
      contains one or more unsupported claims. Treating `pass=True` as "the detector fired"
      makes it natural to gate CI on `pass=False` ("no hallucination detected").

    Use [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] instead if you prefer the
    conventional direction where `pass=True` means the output meets the criterion.
    """

    context: ContextSource
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    score: OutputConfig | Literal[False] = field(default_factory=_default_score_config)
    assertion: OutputConfig | Literal[False] = field(default_factory=_default_assertion_config)

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        resolved_context = _resolve(self.context, ctx)
        judge_inputs = {'question': ctx.inputs, 'context': resolved_context}
        grading_output = await judge_input_output(
            judge_inputs, ctx.output, _HALLUCINATION_RUBRIC, self.model, self.model_settings
        )
        return _emit(self, grading_output, self.score, self.assertion)


@dataclass(repr=False)
class GEval(Evaluator[object, object, object]):
    """G-Eval-style chain-of-thought evaluator (Liu et al., 2023).

    The judge is shown the evaluation `criteria` and a list of explicit `evaluation_steps`,
    produces a short reasoning trace, and emits an integer score within `score_range`.

    !!! note "Simplified G-Eval"
        The paper computes a probability-weighted expectation over score tokens using log-probs.
        We ask the model for a direct integer score instead, trading some correlation with human
        judgment for provider-agnostic simplicity.
    """

    criteria: str
    evaluation_steps: list[str]
    score_range: tuple[int, int] = (1, 5)
    include_input: bool = False
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = None

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        inputs = ctx.inputs if self.include_input else None
        g_eval_output = await judge_g_eval(
            ctx.output,
            self.criteria,
            self.evaluation_steps,
            self.score_range,
            inputs=inputs,
            model=self.model,
            model_settings=self.model_settings,
        )
        return EvaluationReason(value=g_eval_output.score, reason=g_eval_output.reason)


@dataclass(repr=False)
class GembaScore(Evaluator[object, object, object]):
    """GEMBA translation quality evaluator (Kocmi & Federmann, 2023).

    Two variants are supported, matching the published prompts:

    - `'DA'` — Direct Assessment, integer score 0-100.
    - `'SQM'` — Scalar Quality Metrics, integer score 0-6.

    The source text is taken from `ctx.inputs`, the candidate translation from `ctx.output`, and
    the (optional) human reference from `ctx.expected_output`. Pass `source` / `candidate` /
    `reference` callables to override.
    """

    source_lang: str
    target_lang: str
    score_type: Literal['DA', 'SQM'] = 'DA'
    source: ValueSource | None = None
    candidate: ValueSource | None = None
    reference: ValueSource | None = None
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None
    evaluation_name: str | None = None

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        source_text = _resolve(self.source, ctx) if self.source is not None else _stringify_value(ctx.inputs)
        target_text = _resolve(self.candidate, ctx) if self.candidate is not None else _stringify_value(ctx.output)

        reference_text: str | None
        if self.reference is not None:
            reference_text = _resolve(self.reference, ctx)
        elif ctx.expected_output is not None:
            reference_text = _stringify_value(ctx.expected_output)
        else:
            reference_text = None

        if self.score_type == 'DA':
            gemba_output = await judge_gemba_da(
                source_text,
                target_text,
                self.source_lang,
                self.target_lang,
                reference=reference_text,
                model=self.model,
                model_settings=self.model_settings,
            )
        else:
            gemba_output = await judge_gemba_sqm(
                source_text,
                target_text,
                self.source_lang,
                self.target_lang,
                reference=reference_text,
                model=self.model,
                model_settings=self.model_settings,
            )
        return EvaluationReason(value=gemba_output.score, reason=gemba_output.reason)


def _stringify_value(value: object) -> str:
    """Coerce an arbitrary value to a string for GEMBA prompt interpolation."""
    if isinstance(value, str):
        return value
    return str(value)
