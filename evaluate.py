"""
LLM-as-judge evaluator for the BPS School Navigator chatbot.

HOW IT WORKS
------------
This script uses a separate LLM call to score chatbot responses — the same
model that powers the chatbot acts as a "judge" here. For each test case, the
judge is given the conversation history, any retrieved school context, and the
bot's response, then asked to reason step-by-step before assigning a score
(G-Eval style / Chain-of-Thought). Scores are 0–5, normalized to 0.0–1.0.

METRICS
-------
- answer_relevancy   : Does the response address what the user actually asked?
- faithfulness       : Does the response only claim things found in the retrieved
                       school context? (catches hallucinated phone numbers,
                       programs, class sizes, etc.)
- task_completion    : Is the bot making meaningful progress toward helping the
                       family find a school?
- intake_adherence   : Did the bot ask for grade + neighborhood before recommending
                       specific schools? (enforces the 3-phase intake protocol)

SETUP
-----
Requires a .env file in the repo root with your HuggingFace token:
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

USAGE
-----
  python evaluate.py                                      # run all 5 built-in test cases
  python evaluate.py --case hallucination                 # run one case by name
  python evaluate.py --metric faithfulness                # run one metric across all cases
  python evaluate.py --file conversation_logs/abc123.json # evaluate full conversation holistically

Available test cases:
  good_intake                  — bot correctly collects info before recommending
  good_faithful                — recommendation grounded in real context data
  hallucination                — bot makes up facts not in the context (should score low)
  skips_intake                 — bot recommends without asking grade/neighborhood (should score low)
  mixed_helpful_no_disclaimer  — helpful but missing the lottery disclaimer

TO ADD A NEW TEST CASE
----------------------
Add a TestCase to the list returned by build_test_cases(). Each case needs:
  - name            : a short unique identifier
  - conversation    : list of Turn("user"/"bot", "message") objects
  - retrieved_context: the school data that would have been retrieved (paste
                       rows from bps_clean.csv as plain text, or "" if none)
  - response        : the bot response you want to evaluate
  - notes           : (optional) what you expect the scores to look like

OUTPUT
------
  Prints a per-case, per-metric score table to stdout.
  Writes full judge reasoning to evaluate_results.json.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict

from huggingface_hub import InferenceClient
from config import BASE_MODEL, HF_TOKEN


# ── Metric definitions ────────────────────────────────────────────────────────

# Criteria for evaluating a single response (used by built-in test cases)
METRICS = {
    "answer_relevancy": {
        "description": "Answer Relevancy",
        "criteria": (
            "Does the response directly and completely address what the user asked "
            "in their most recent message? A relevant response stays on topic and "
            "does not answer a different question or provide excessive unrelated information."
        ),
    },
    "faithfulness": {
        "description": "Faithfulness (Anti-Hallucination)",
        "criteria": (
            "Does the response only state facts that are explicitly present in the "
            "retrieved school context? Penalize any specific claims (phone numbers, "
            "class sizes, program details, addresses, schedules) that are NOT found "
            "in the context. If no context is provided, the response should avoid "
            "stating specific school facts altogether and instead say it doesn't have "
            "that information."
        ),
    },
    "task_completion": {
        "description": "Task Completion",
        "criteria": (
            "Is the bot making meaningful progress toward the goal of helping this "
            "family find and enroll in a suitable Boston public school? A high score "
            "means the response moves the conversation forward (collects needed info, "
            "narrows options, or provides actionable next steps). A low score means "
            "the response stalls, repeats itself, or leaves the user with nothing useful."
        ),
    },
    "intake_adherence": {
        "description": "Intake Adherence",
        "criteria": (
            "Does the bot follow the required intake protocol? Before recommending "
            "specific schools, it must first collect: (1) the child's grade level, "
            "and (2) the family's neighborhood or zip code. If the bot recommends "
            "specific schools without first obtaining both of these, that is a "
            "protocol violation. If the bot is still in the intake phase and asking "
            "the right questions, that should score highly."
        ),
    },
}

# Criteria for evaluating a full conversation holistically (used with --file)
HOLISTIC_METRICS = {
    "answer_relevancy": {
        "description": "Answer Relevancy",
        "criteria": (
            "Across the full conversation, did the bot consistently address what "
            "users asked? Consider all of the bot's responses collectively — did it "
            "stay on topic, avoid tangents, and respond to what was actually asked?"
        ),
    },
    "faithfulness": {
        "description": "Faithfulness (Anti-Hallucination)",
        "criteria": (
            "Across ALL of the bot's responses in this conversation, did it only "
            "claim facts explicitly present in the retrieved school context? "
            "Any specific claim not found in the context — including school names, "
            "addresses, phone numbers, enrollment figures, program details, class sizes, "
            "schedules, or registration deadlines — counts as a hallucination and must "
            "result in a low score. If no context was provided, the bot must not state "
            "any specific school facts at all; doing so is an automatic score of 0-1. "
            "A score of 4 or 5 requires zero hallucinated facts across the entire conversation."
        ),
    },
    "task_completion": {
        "description": "Task Completion",
        "criteria": (
            "Did the conversation as a whole successfully help the family find and "
            "understand their school options? A high score means the bot collected "
            "the right information, provided useful recommendations, and left the "
            "family with clear next steps. A low score means the conversation ended "
            "without meaningful progress."
        ),
    },
    "intake_adherence": {
        "description": "Intake Adherence",
        "criteria": (
            "Did the bot properly collect both (1) the child's grade level and "
            "(2) the family's neighborhood or zip code before recommending specific "
            "schools? Review the full conversation — if the bot made recommendations "
            "before obtaining both pieces of information, that is a protocol violation."
        ),
    },
}

JUDGE_PROMPT = """\
You are evaluating a response from the BPS School Navigator chatbot, which helps \
Boston families find public schools for their children.

=== CONVERSATION HISTORY ===
{conversation}

=== RETRIEVED SCHOOL CONTEXT ===
{context}

=== RESPONSE TO EVALUATE ===
{response}

=== EVALUATION TASK ===
Metric: {metric_name}
Criteria: {criteria}

Think step by step about whether this response meets the criteria above.
Consider the full conversation history and the retrieved context when making your judgment.

After your reasoning, give a score from 0 to 5:
  0 = Completely fails the criteria
  1 = Very poor
  2 = Below expectations
  3 = Acceptable
  4 = Good
  5 = Excellent / fully meets criteria

Respond in exactly this format:
Reasoning: <your step-by-step reasoning>
Score: <integer from 0 to 5>
"""

HOLISTIC_JUDGE_PROMPT = """\
You are evaluating the BPS School Navigator chatbot's overall performance across \
a complete conversation with a Boston family looking for public schools.

=== FULL CONVERSATION ===
{conversation}

=== RETRIEVED SCHOOL CONTEXT ===
{context}

=== EVALUATION TASK ===
Metric: {metric_name}
Criteria: {criteria}

Evaluate the bot's overall behavior across the ENTIRE conversation based on the \
criteria above. Consider all of the bot's responses collectively, not just one turn.

After your reasoning, give a score from 0 to 5:
  0 = Completely fails the criteria
  1 = Very poor
  2 = Below expectations
  3 = Acceptable
  4 = Good
  5 = Excellent / fully meets criteria

Respond in exactly this format:
Reasoning: <your step-by-step reasoning>
Score: <integer from 0 to 5>
"""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str   # "user" or "bot"
    content: str


@dataclass
class TestCase:
    name: str
    conversation: list[Turn]
    retrieved_context: str          # school data that was retrieved
    response: str                   # single response to evaluate (empty if holistic=True)
    notes: str = ""                 # human notes on what to expect
    holistic: bool = False          # if True, evaluate the full conversation instead of a single response


@dataclass
class MetricResult:
    score_raw: int          # 0–5
    score_normalized: float # 0.0–1.0
    reasoning: str


@dataclass
class EvalResult:
    case_name: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)

    def average_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.score_normalized for m in self.metrics.values()) / len(self.metrics)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, model_id: str = BASE_MODEL):
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)

    def _format_conversation(self, turns: list[Turn]) -> str:
        lines = []
        for t in turns:
            label = "User" if t.role == "user" else "Bot"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    def _parse_score(self, text: str) -> tuple[int, str]:
        """Extract score and reasoning from judge output."""
        score_match = re.search(r"Score:\s*([0-5])", text)
        score = int(score_match.group(1)) if score_match else 0

        reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=Score:|$)", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()

        return score, reasoning

    def evaluate_metric(
        self,
        metric_key: str,
        conversation: list[Turn],
        retrieved_context: str,
        response: str,
        holistic: bool = False,
    ) -> MetricResult:
        metric = (HOLISTIC_METRICS if holistic else METRICS)[metric_key]
        context_text = retrieved_context if retrieved_context.strip() else "(no context retrieved)"

        if holistic:
            prompt = HOLISTIC_JUDGE_PROMPT.format(
                conversation=self._format_conversation(conversation),
                context=context_text,
                metric_name=metric["description"],
                criteria=metric["criteria"],
            )
        else:
            prompt = JUDGE_PROMPT.format(
                conversation=self._format_conversation(conversation),
                context=context_text,
                response=response,
                metric_name=metric["description"],
                criteria=metric["criteria"],
            )

        messages = [{"role": "user", "content": prompt}]
        output = self.client.chat_completion(messages=messages, max_tokens=512)
        raw_text = output.choices[0].message.content

        score_raw, reasoning = self._parse_score(raw_text)
        return MetricResult(
            score_raw=score_raw,
            score_normalized=score_raw / 5.0,
            reasoning=reasoning,
        )

    def evaluate_case(self, case: TestCase, metrics: list[str] | None = None) -> EvalResult:
        metrics = metrics or list(METRICS.keys())
        result = EvalResult(case_name=case.name)

        for metric_key in metrics:
            print(f"  Scoring {METRICS[metric_key]['description']}...")
            result.metrics[metric_key] = self.evaluate_metric(
                metric_key=metric_key,
                conversation=case.conversation,
                retrieved_context=case.retrieved_context,
                response=case.response,
                holistic=case.holistic,
            )

        return result


# ── Test cases ────────────────────────────────────────────────────────────────

def build_test_cases() -> list[TestCase]:
    """
    A set of test cases ranging from good to bad responses.
    Add more here as the chatbot develops.
    """

    # Context representing what would be retrieved from bps_clean.csv
    hernandez_context = """\
School: Hernandez K-8 School
Address: 61 School Street, Roxbury, MA 02119
Phone: 617-635-8428
Grades Served: PK - 08
Enrollment: 410
Student-Teacher Ratio: 11.2 to 1
% English Learners: 38.4
% Students with Disabilities: 19.1
% Low Income: 91.3
SPED Programs: Mild Intellectual Disabilities; Specific Learning Disabilities
"""

    # --- GOOD: Proper intake before recommending ---
    good_intake = TestCase(
        name="good_intake",
        conversation=[
            Turn("user", "Hi, I'm looking for a public school in Boston for my child who will be starting kindergarten next year."),
        ],
        retrieved_context="",
        response=(
            "Hello! I'd be happy to help you find the right school. To show you the schools "
            "available to your family, I first need a couple of details:\n\n"
            "1. What neighborhood or zip code do you live in? (This determines which schools "
            "your child is eligible for under Boston's home-based assignment system.)\n"
            "2. Does your child have any specific needs, like special education services or "
            "interest in a dual-language program?\n\n"
            "Once I know your neighborhood, I can pull up the schools on your eligibility list."
        ),
        notes="Bot correctly asks for neighborhood before recommending. Should score high on intake_adherence.",
    )

    # --- GOOD: Faithful recommendation using only context data ---
    good_faithful = TestCase(
        name="good_faithful",
        conversation=[
            Turn("user", "Hi, looking for a school for my kindergartner."),
            Turn("bot", "What neighborhood do you live in, and does your child have any specific needs?"),
            Turn("user", "We're in Roxbury. My child is interested in learning Spanish."),
        ],
        retrieved_context=hernandez_context,
        response=(
            "Based on your location in Roxbury, the **Hernandez K-8 School** looks like a strong match. "
            "Here's what I have on file:\n\n"
            "- Grades: PK–8, so your child can stay through middle school\n"
            "- Enrollment: ~410 students\n"
            "- Phone: 617-635-8428\n"
            "- Address: 61 School Street, Roxbury\n\n"
            "A note on enrollment: BPS uses a lottery-based system, so ranking this school highly "
            "on your application gives you the best chance, but is not a guarantee of a seat. "
            "Would you like information on the registration process?"
        ),
        notes="Bot uses only data from context and includes lottery disclaimer. Should score high on faithfulness.",
    )

    # --- BAD: Hallucination — bot makes up facts not in context ---
    hallucination = TestCase(
        name="hallucination",
        conversation=[
            Turn("user", "Tell me about the Hernandez School."),
        ],
        retrieved_context=hernandez_context,
        response=(
            "The Hernandez K-8 School is excellent! They have a 50/50 Spanish-English dual language "
            "program with an average class size of 22 students. Tours are offered every Tuesday and "
            "Thursday at 9 AM. You can email them at hernandez@bostonpublicschools.org or call "
            "(617) 635-XXXX. They also have a very active parent-teacher organization and "
            "after-school programs until 6 PM."
        ),
        notes=(
            "Bot fabricated: dual-language program, class size, tour schedule, email. "
            "Should score very low on faithfulness."
        ),
    )

    # --- BAD: Skips intake and recommends without knowing neighborhood or grade ---
    skips_intake = TestCase(
        name="skips_intake",
        conversation=[
            Turn("user", "What are the best schools in Boston?"),
        ],
        retrieved_context="",
        response=(
            "Boston has many great public schools! Here are some of the top ones:\n"
            "1. Boston Latin School — the oldest public school in America\n"
            "2. Boston Arts Academy — great for creative students\n"
            "3. Hernandez K-8 — strong dual language program\n"
            "4. Eliot K-8 in the North End — very popular\n\n"
            "Any of these interest you?"
        ),
        notes=(
            "Bot jumps straight to recommendations without asking grade or neighborhood. "
            "Should score low on intake_adherence."
        ),
    )

    # --- MIXED: Relevant and helpful, but slightly off protocol ---
    mixed_case = TestCase(
        name="mixed_helpful_no_disclaimer",
        conversation=[
            Turn("user", "We live in Jamaica Plain. My daughter is going into 1st grade and we want a school with Spanish."),
            Turn("bot", "Great — let me look up schools in Jamaica Plain for grades K–1 with Spanish programs."),
            Turn("user", "Yes please."),
        ],
        retrieved_context=hernandez_context,
        response=(
            "Based on your location in Jamaica Plain, here's a school that may be a good fit:\n\n"
            "**Hernandez K-8 School** (Roxbury — about 1.5 miles from JP)\n"
            "- Grades: PK–8\n"
            "- Phone: 617-635-8428\n"
            "- Enrollment: ~410 students\n\n"
            "This school serves students from 1st grade and beyond. Would you like to know how to apply?"
        ),
        notes=(
            "Response is helpful and mostly faithful, but omits the lottery disclaimer. "
            "Should score moderately — good on relevancy/faithfulness, lower on task_completion "
            "for missing the disclaimer."
        ),
    )

    return [good_intake, good_faithful, hallucination, skips_intake, mixed_case]


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult]) -> None:
    header = f"{'Case':<35} {'AnswRel':>8} {'Faith':>8} {'TaskCmp':>8} {'Intake':>8} {'Avg':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        scores = {k: r.metrics[k].score_normalized if k in r.metrics else None for k in METRICS}
        def fmt(v):
            return f"{v:.2f}" if v is not None else "  --  "
        print(
            f"{r.case_name:<35} "
            f"{fmt(scores['answer_relevancy']):>8} "
            f"{fmt(scores['faithfulness']):>8} "
            f"{fmt(scores['task_completion']):>8} "
            f"{fmt(scores['intake_adherence']):>8} "
            f"{r.average_score():>8.2f}"
        )

    print("=" * len(header))
    print()


def save_results(results: list[EvalResult], path: str = "evaluate_results.json") -> None:
    data = []
    for r in results:
        entry = {"case": r.case_name, "average": r.average_score(), "metrics": {}}
        for k, m in r.metrics.items():
            entry["metrics"][k] = asdict(m)
        data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Full reasoning saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_cases_from_file(path: str) -> list[TestCase]:
    """
    Load a conversation log file saved by app.py as a single holistic test case.

    The full conversation (all user and bot turns) is passed to the judge, which
    evaluates the bot's overall behavior across the entire conversation rather than
    scoring each response individually.

    The file should be a JSON object with:
      - conversation : list of {"role": "user"/"assistant"/"bot", "content": "..."}
      - retrieved_context : string (can be empty)
      - name : (optional) identifier
    """
    with open(path) as f:
        data = json.load(f)

    name = data.get("name", os.path.splitext(os.path.basename(path))[0])
    turns = [Turn(role=t["role"], content=t["content"]) for t in data["conversation"]]

    return [TestCase(
        name=name,
        conversation=turns,
        retrieved_context=data.get("retrieved_context", ""),
        response="",
        holistic=True,
    )]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the BPS chatbot.")
    parser.add_argument("--case", type=str, default=None, help="Run a single named built-in test case.")
    parser.add_argument("--metric", type=str, default=None, help="Run a single metric only.")
    parser.add_argument("--file", type=str, default=None, help="Path to a conversation log file from app.py.")
    args = parser.parse_args()

    evaluator = Evaluator()

    if args.file:
        test_cases = load_cases_from_file(args.file)
    else:
        test_cases = build_test_cases()
        if args.case:
            test_cases = [c for c in test_cases if c.name == args.case]
            if not test_cases:
                print(f"No test case named '{args.case}'. Available: {[c.name for c in build_test_cases()]}")
                return

    metrics_to_run = [args.metric] if args.metric and args.metric in METRICS else None

    results = []
    for case in test_cases:
        print(f"\nEvaluating: {case.name}")
        if case.notes:
            print(f"  ({case.notes})")
        result = evaluator.evaluate_case(case, metrics=metrics_to_run)
        results.append(result)

    print_report(results)
    save_results(results)


if __name__ == "__main__":
    main()
