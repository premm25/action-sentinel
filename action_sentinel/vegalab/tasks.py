"""Public evaluation tasks for the Vega lab.

Selection rule (documented in docs/vegalab/DATASETS.md): exclude every task family the shipped
VegaML adapters were trained on (LocalLLaMA/typed-decisions: apibank, appliances, banking77,
clinc150, esci, when2call) and the datasets used in VegaML's own fine-tuning example (emotion,
ag_news). VegaML's BASE engine training corpus is not publicly documented, so leakage into the base
engine can't be ruled out for any public dataset. Results carry that caveat.

Question wording and option descriptions are ours. They are fixed here and shown identically to
every system (VegaML, backbone baseline, deterministic baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .contracts import CandidateOption, DecisionQuestion


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_dataset: str
    hf_config: str
    eval_split: str
    fit_split: str                       # source of calibration and dev examples (disjoint from eval)
    qtype: str
    instructions: str
    options: list[tuple[str, str]]       # (label, description) in label-index order
    gold: Callable[[dict[str, Any]], str]
    state: Callable[[dict[str, Any]], Any]
    group: str
    license: str
    difficulty: str = "standard"
    notes: str = ""
    dynamic_labels: str | None = None    # column whose distinct values form the label space
    extra: dict[str, Any] = field(default_factory=dict)

    def question(self, options: list[tuple[str, str]] | None = None) -> DecisionQuestion:
        opts = options if options is not None else self.options
        return DecisionQuestion(question_id="answer", type=self.qtype, instructions=self.instructions,
                                options=[CandidateOption(label=lab, description=desc) for lab, desc in opts])


def _idx(names: list[str]):
    def gold(row: dict[str, Any]) -> str:
        i = int(row["label"])
        if not 0 <= i < len(names):   # e.g. -1 marks a hidden label; never wrap around to names[-1]
            raise ValueError("label %r out of range" % row["label"])
        return names[i]
    return gold


DBPEDIA = [("Company", "a company or business"), ("EducationalInstitution", "a school, college or university"),
           ("Artist", "an artist, musician or writer"), ("Athlete", "an athlete or sports person"),
           ("OfficeHolder", "a politician or office holder"), ("MeanOfTransportation", "a vehicle, ship or aircraft"),
           ("Building", "a building or structure"), ("NaturalPlace", "a mountain, river, lake or other natural place"),
           ("Village", "a village or small settlement"), ("Animal", "an animal species"),
           ("Plant", "a plant species"), ("Album", "a music album"), ("Film", "a film"),
           ("WrittenWork", "a book, journal or other written work")]

TASKS: dict[str, TaskSpec] = {t.name: t for t in [
    TaskSpec(
        "sst2", "stanfordnlp/sst2", "default", "validation", "train", "choice",
        "What sentiment does this movie review sentence express?",
        [("negative", "a negative opinion"), ("positive", "a positive opinion")],
        _idx(["negative", "positive"]), lambda r: r["sentence"], "sentiment", "unknown-see-card"),
    TaskSpec(
        "dbpedia14", "fancyzhx/dbpedia_14", "dbpedia_14", "test", "train", "choice",
        "Which kind of entity does this encyclopedia article describe?", DBPEDIA,
        _idx([d[0] for d in DBPEDIA]), lambda r: {"title": r["title"], "article": r["content"]}, "topic",
        "cc-by-sa-3.0"),
    TaskSpec(
        "massive_intent_en", "mteb/amazon_massive_intent", "en", "test", "train", "choice",
        "Which voice-assistant intent does this utterance express?", [],
        lambda r: r["label_text"], lambda r: r["text"], "intent", "cc-by-4.0", difficulty="wide_label_space",
        dynamic_labels="label_text", notes="60 intents; label space taken from the fit split"),
    TaskSpec(
        "glue_rte", "nyu-mll/glue", "rte", "validation", "train", "boolean",
        "Does the first sentence entail the second sentence?",
        [("false", "the first sentence does not entail the second"), ("true", "the first sentence entails the second")],
        lambda r: "true" if int(r["label"]) == 0 else "false",
        lambda r: {"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]}, "entailment", "other-see-card"),
    TaskSpec(
        "boolq", "google/boolq", "default", "validation", "train", "boolean",
        "According to the passage, is the answer to the question yes?",
        [("false", "no"), ("true", "yes")],
        lambda r: "true" if r["answer"] in (True, "true", "True", 1) else "false",
        lambda r: {"passage": r["passage"], "question": r["question"]}, "reading", "cc-by-sa-3.0"),
    TaskSpec(
        "superglue_cb", "aps/super_glue", "cb", "validation", "train", "choice",
        "Does the premise entail the hypothesis, contradict it, or neither?",
        [("entailment", "the premise entails the hypothesis"),
         ("contradiction", "the premise contradicts the hypothesis"),
         ("neutral", "neither: the premise does not settle it")],
        _idx(["entailment", "contradiction", "neutral"]),
        lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "entailment", "other-see-card",
        notes="validation has only 56 examples"),
    TaskSpec(
        "prompt_injections", "deepset/prompt-injections", "default", "test", "train", "boolean",
        "Is this input a prompt-injection attempt?",
        [("false", "an ordinary request or statement"),
         ("true", "it tries to override the instructions the system was given")],
        lambda r: "true" if int(r["label"]) == 1 else "false", lambda r: r["text"], "safety", "apache-2.0",
        notes="mixed languages (English and German)"),
    TaskSpec(
        "yelp_review_stars", "Yelp/yelp_review_full", "yelp_review_full", "test", "train", "score",
        "How many stars does this review give?",
        [("0", "1 star: very negative"), ("1", "2 stars: negative"), ("2", "3 stars: mixed or neutral"),
         ("3", "4 stars: positive"), ("4", "5 stars: very positive")],
        lambda r: str(int(r["label"])), lambda r: r["text"], "sentiment", "yelp-dataset-terms",
        difficulty="ordinal"),
    TaskSpec(
        "anli_r3", "facebook/anli", "plain_text", "test_r3", "train_r3", "choice",
        "Does the premise entail the hypothesis, contradict it, or neither?",
        [("entailment", "the premise entails the hypothesis"), ("neutral", "neither: the premise does not settle it"),
         ("contradiction", "the premise contradicts the hypothesis")],
        _idx(["entailment", "neutral", "contradiction"]),
        lambda r: {"premise": r["premise"], "hypothesis": r["hypothesis"]}, "entailment", "cc-by-nc-4.0",
        difficulty="adversarial", notes="adversarially collected NLI, round 3"),
]}

EXCLUDED = {
    "banking77": "VegaML typed_decisions adapter training workflow",
    "clinc150": "VegaML typed_decisions adapter training workflow",
    "apibank": "VegaML typed_decisions adapter training workflow",
    "appliances": "VegaML typed_decisions adapter training workflow",
    "esci": "VegaML typed_decisions adapter training workflow",
    "when2call": "VegaML typed_decisions adapter training workflow",
    "emotion": "used in VegaML's fine-tuning example",
    "ag_news": "used as gate negatives in VegaML's fine-tuning example",
    "trec": "not available as parquet (loading script only)",
}

# RQ5: same dbpedia14 test examples, varying the number of candidate options (gold plus random distractors).
OPTION_COUNT_VARIANTS = (2, 4, 8)
