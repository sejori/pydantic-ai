# Standard Quality Metrics

Pydantic Evals ships a small curated pack of LLM-backed evaluators whose names and rubrics
track widely-used evaluation frameworks — [Ragas](https://github.com/explodinggradients/ragas),
G-Eval (Liu et al., 2023) and GEMBA (Kocmi & Federmann, 2023).

These evaluators are thin wrappers over [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] and the
[`judge_*`][pydantic_evals.evaluators.llm_as_a_judge] helpers. You can always write your own rubric
against `LLMJudge` — the point of this pack is to provide recognisable names and sensible defaults
for the quality dimensions users most often reach for. Each evaluator takes the same `model`,
`model_settings`, `score` and `assertion` arguments as `LLMJudge`, so they drop into existing
evaluation suites without ceremony.

!!! tip "Mix and match"
    Nothing stops you from combining these curated evaluators with a custom
    [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] or your own domain-specific evaluator in the
    same dataset. Many users end up with one or two curated metrics plus case-specific rubrics.

## Faithfulness (Ragas)

Faithfulness checks that every factual claim in the output is grounded in a provided retrieval
context. Unsupported, contradicted, or fabricated claims cause the assertion to fail; the score
is the fraction of claims that are supported.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Faithfulness

dataset = Dataset(
    name='faithfulness_demo',
    cases=[
        Case(
            inputs='Where is the Eiffel Tower?',
            metadata={'context': 'The Eiffel Tower is in Paris, France.'},
        ),
    ],
    evaluators=[
        Faithfulness(context=lambda ctx: (ctx.metadata or {})['context']),
    ],
)
```

The `context` argument accepts either a fixed string (useful in tests) or a callable that pulls
the context out of the [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] (useful for
real RAG pipelines where each case has its own retrieved passages).

## Answer Relevance (Ragas)

[`AnswerRelevance`][pydantic_evals.evaluators.AnswerRelevance] judges whether the output directly
addresses the question in the input, without padding or tangents.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import AnswerRelevance

dataset = Dataset(
    name='answer_relevance_demo',
    cases=[Case(inputs='What is the speed of light?')],
    evaluators=[AnswerRelevance()],
)
```

By default the question is read from `ctx.inputs`. Pass `question=lambda ctx: ...` to extract the
question from a richer input structure.

## Context Precision (Ragas)

[`ContextPrecision`][pydantic_evals.evaluators.ContextPrecision] estimates how much of a retrieved
context is actually relevant to the question — a low score flags retrievers that drown the
downstream model in noise.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ContextPrecision

retrieved_context = (
    'Paris is the capital of France. The Seine flows through Paris. '
    'Lyon is a city in France. Bordeaux is a city in France.'
)

dataset = Dataset(
    name='context_precision_demo',
    cases=[Case(inputs='What is the capital of France?')],
    evaluators=[ContextPrecision(context=retrieved_context)],
)
```

## Context Recall (Ragas)

[`ContextRecall`][pydantic_evals.evaluators.ContextRecall] checks whether the retrieved context
contains enough information to produce the ground-truth answer. This needs an `expected_output`
(or a `ground_truth=` callable); it returns an empty result if neither is available.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ContextRecall

dataset = Dataset(
    name='context_recall_demo',
    cases=[
        Case(
            inputs='Who wrote Pride and Prejudice?',
            expected_output='Jane Austen wrote Pride and Prejudice in 1813.',
        ),
    ],
    evaluators=[
        ContextRecall(context='Pride and Prejudice is a classic English novel.'),
    ],
)
```

## Hallucination

[`Hallucination`][pydantic_evals.evaluators.Hallucination] is the sibling of `Faithfulness`,
framed as a detector:

- `score` is a groundedness score — **higher is better** (1.0 = fully grounded, 0.0 = fully
  hallucinated).
- The assertion fires (`pass=True`) when the detector **does** find hallucinated claims. This is
  the opposite of most assertions, and deliberate: CI can gate on `pass=False` to mean "no
  hallucination detected".

If you prefer the conventional direction where `pass=True` means "the output meets the criterion",
use [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] instead.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Hallucination

dataset = Dataset(
    name='hallucination_demo',
    cases=[Case(inputs='Summarise the article.')],
    evaluators=[Hallucination(context='The article discusses Mars exploration...')],
)
```

## G-Eval (Liu et al., 2023)

[`GEval`][pydantic_evals.evaluators.GEval] implements a chain-of-thought evaluator: you provide
the aspect being evaluated (`criteria`) and a list of explicit `evaluation_steps`, and the judge
returns a reasoning trace plus an integer score in `score_range`.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GEval

dataset = Dataset(
    name='g_eval_demo',
    cases=[Case(inputs='Explain how black holes form.')],
    evaluators=[
        GEval(
            criteria='coherence',
            evaluation_steps=[
                'Read the output carefully.',
                'Check that each sentence follows logically from the previous one.',
                'Assign a score from 1 (incoherent) to 5 (fully coherent).',
            ],
            include_input=True,
        ),
    ],
)
```

!!! note "Simplified G-Eval"
    The published G-Eval method computes a probability-weighted expectation over score tokens
    using the judge model's log-probs. Pydantic Evals asks the model for a direct integer score
    instead, trading a small amount of correlation with human judgment for provider-agnostic
    simplicity. See Liu et al., 2023, "G-Eval: NLG Evaluation using GPT-4 with Better Human
    Alignment".

## GEMBA (Kocmi & Federmann, 2023)

[`GembaScore`][pydantic_evals.evaluators.GembaScore] is a machine-translation quality evaluator
using the GEMBA prompts. Two variants are supported, matching the published prompts:

- `score_type='DA'` — Direct Assessment, integer score **0-100**.
- `score_type='SQM'` — Scalar Quality Metrics, integer score **0-6**.

By default the source text is read from `ctx.inputs`, the candidate translation from `ctx.output`,
and an optional human reference from `ctx.expected_output`. Pass `source`, `candidate` or
`reference` callables to override.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GembaScore

dataset = Dataset(
    name='gemba_demo',
    cases=[
        Case(
            inputs='Hello, world!',
            expected_output='Bonjour, le monde !',
        ),
    ],
    evaluators=[
        GembaScore(source_lang='English', target_lang='French'),
    ],
)
```

See Kocmi & Federmann, 2023, "Large Language Models Are State-of-the-Art Evaluators of Translation
Quality".

## Picking the right tool

| Need | Use |
| --- | --- |
| Is the output supported by my retrieved context? | [`Faithfulness`][pydantic_evals.evaluators.Faithfulness] |
| Did the model hallucinate anything? (gate on `pass=False`) | [`Hallucination`][pydantic_evals.evaluators.Hallucination] |
| Does the answer actually address the question? | [`AnswerRelevance`][pydantic_evals.evaluators.AnswerRelevance] |
| Is my retriever surfacing relevant passages? | [`ContextPrecision`][pydantic_evals.evaluators.ContextPrecision] |
| Is my retriever returning enough information? | [`ContextRecall`][pydantic_evals.evaluators.ContextRecall] |
| Score a quality dimension on an integer scale with explicit CoT steps | [`GEval`][pydantic_evals.evaluators.GEval] |
| Score a translation against a source (and optional reference) | [`GembaScore`][pydantic_evals.evaluators.GembaScore] |
| Something bespoke | [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] with a custom rubric |

For bringing in the exact upstream implementations of these and other frameworks, see
[Framework Integrations](framework-integrations.md).
