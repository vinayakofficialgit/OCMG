"""Synthetic multi-session dialogue generator.

The generator exists so the pipeline can be exercised and defended without a
benchmark download, and so that the properties governance is supposed to
exploit can be controlled and varied rather than hoped for. Four are injected
deliberately:

  low-value chatter   most turns carry no retrievable fact, so an append-all
                      store fills with material that never supports an answer
  redundancy          facts are restated across sessions, so the merge path is
                      genuinely exercised
  contradiction       a later session updates an earlier fact, so retrieving
                      the stale version produces a wrong answer
  distractors         facts that resemble the target but do not answer it

Questions span the five LoCoMo categories. Every question carries the ids of the
facts that support it, which is what makes the outcome signal well defined.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from ..types import Question, Turn

SUBJECTS = [
    ("coffee", "flat white", "cortado"), ("commute", "bicycle", "train"),
    ("gym", "Tuesday", "Thursday"), ("laptop", "Dell XPS", "MacBook Air"),
    ("cat", "Mochi", "Pepper"), ("city", "Bristol", "Leeds"),
    ("language class", "Portuguese", "Japanese"), ("thesis", "chapter 3", "chapter 4"),
    ("car", "Corolla", "Civic"), ("phone", "Pixel 8", "iPhone 14"),
    ("holiday", "Lisbon", "Porto"), ("dentist", "Dr Ahmed", "Dr Rowe"),
    ("bank", "Monzo", "Starling"), ("flat", "Elm Road", "Bay Street"),
    ("teammate", "Priya", "Marcus"), ("running shoes", "Brooks", "Asics"),
]

CHATTER = [
    "That reminds me, the weather has been strange all week.",
    "I keep meaning to sort out the cupboard under the stairs.",
    "Anyway, it is not a big deal either way.",
    "I watched a documentary last night, it was quite long.",
    "The neighbours were doing DIY again this morning.",
    "I should probably drink more water during the day.",
    "There was a queue at the post office for twenty minutes.",
    "I have been meaning to reply to that email for days.",
    "The bus was late again, which is becoming a pattern.",
    "I tried a new recipe and it came out reasonably well.",
    "My phone battery has been draining faster recently.",
    "Someone left a bicycle outside the building all week.",
]

ACK = ["Got it.", "Understood.", "Noted.", "Thanks for letting me know.",
       "That makes sense.", "Okay."]

DAY = 86400.0


def _fact_sentence(topic: str, value: str, updated: bool = False) -> str:
    if updated:
        return f"Actually I switched my {topic} to {value} last week."
    return f"My {topic} is {value}."


def generate_conversation(
    conversation_id: str,
    rng: random.Random,
    n_sessions: int = 8,
    turns_per_session: int = 14,
    n_facts: int = 10,
    update_prob: float = 0.35,
    start_time: float = 1_700_000_000.0,
) -> Tuple[List[Turn], List[Question], Dict[str, str]]:
    subjects = rng.sample(SUBJECTS, k=min(n_facts, len(SUBJECTS)))
    turns: List[Turn] = []
    facts: Dict[str, str] = {}       # fact_id -> current answer value
    fact_topic: Dict[str, str] = {}
    superseded: Dict[str, str] = {}  # stale fact_id -> topic

    turn_counter = 0
    fact_queue = list(enumerate(subjects))
    rng.shuffle(fact_queue)

    for s in range(n_sessions):
        session_time = start_time + s * 7 * DAY
        # One or two facts land per session; the rest of the session is noise.
        n_new = 1 if s > 0 and rng.random() < 0.5 else 2
        landing = [fact_queue.pop() for _ in range(min(n_new, len(fact_queue)))]

        for i in range(turns_per_session):
            speaker = "user" if i % 2 == 0 else "assistant"
            ts = session_time + i * 300.0
            fact_id = None

            if speaker == "user" and landing and i in (2, 6):
                idx, (topic, v1, v2) = landing.pop()
                fid = f"{conversation_id}:f{idx}"
                text = _fact_sentence(topic, v1)
                facts[fid] = v1
                fact_topic[fid] = topic
                fact_id = fid
            elif (
                speaker == "user"
                and s >= 3
                and rng.random() < update_prob
                and facts
                and i == 8
            ):
                # Contradiction / update: a later session supersedes an earlier
                # fact. Retrieving the stale record now yields a wrong answer.
                fid = rng.choice(list(facts.keys()))
                topic = fact_topic[fid]
                pair = next((p for p in SUBJECTS if p[0] == topic), None)
                if pair:
                    new_val = pair[2] if facts[fid] == pair[1] else pair[1]
                    stale_id = f"{fid}#stale{s}"
                    superseded[stale_id] = topic
                    text = _fact_sentence(topic, new_val, updated=True)
                    facts[fid] = new_val
                    fact_id = fid
                else:
                    text = rng.choice(CHATTER)
            elif speaker == "user":
                text = rng.choice(CHATTER)
            else:
                text = rng.choice(ACK) if rng.random() < 0.45 else (
                    "I see what you mean, that sounds reasonable."
                )

            turns.append(
                Turn(
                    session_id=s,
                    turn_index=turn_counter,
                    speaker=speaker,
                    text=text,
                    timestamp=ts,
                    fact_id=fact_id,
                )
            )
            turn_counter += 1

    # ---- questions -------------------------------------------------------
    questions: List[Question] = []
    live = list(facts.items())
    rng.shuffle(live)
    ask_time = start_time + (n_sessions + 1) * 7 * DAY

    for j, (fid, value) in enumerate(live):
        topic = fact_topic[fid]
        questions.append(
            Question(
                conversation_id=conversation_id,
                question_id=f"{conversation_id}:q{j}",
                text=f"What is my {topic}?",
                answer=value,
                category="single_hop",
                supporting_fact_ids=[fid],
                asked_at=ask_time,
            )
        )

    # multi-hop: two facts required
    for j in range(min(3, len(live) // 2)):
        a, b = live[2 * j], live[2 * j + 1]
        questions.append(
            Question(
                conversation_id=conversation_id,
                question_id=f"{conversation_id}:mh{j}",
                text=f"What are my {fact_topic[a[0]]} and my {fact_topic[b[0]]}?",
                answer=f"{a[1]} and {b[1]}",
                category="multi_hop",
                supporting_fact_ids=[a[0], b[0]],
                asked_at=ask_time,
            )
        )

    # temporal: requires the most recent value, i.e. the update not the original
    for j, (fid, value) in enumerate(live[:3]):
        questions.append(
            Question(
                conversation_id=conversation_id,
                question_id=f"{conversation_id}:tm{j}",
                text=f"What is my {fact_topic[fid]} most recently?",
                answer=value,
                category="temporal",
                supporting_fact_ids=[fid],
                asked_at=ask_time,
            )
        )

    # adversarial: nothing in memory supports an answer
    for j, topic in enumerate(["passport number", "blood type", "first employer"]):
        questions.append(
            Question(
                conversation_id=conversation_id,
                question_id=f"{conversation_id}:adv{j}",
                text=f"What is my {topic}?",
                answer="unable to determine from memory",
                category="adversarial",
                supporting_fact_ids=[],
                asked_at=ask_time,
            )
        )

    return turns, questions, facts


def generate_dataset(
    n_conversations: int = 20,
    seed: int = 0,
    n_sessions: int = 8,
    turns_per_session: int = 14,
    n_facts: int = 10,
) -> List[Tuple[str, List[Turn], List[Question]]]:
    rng = random.Random(seed)
    out = []
    for c in range(n_conversations):
        cid = f"conv{c:03d}"
        turns, questions, _ = generate_conversation(
            cid, rng, n_sessions=n_sessions,
            turns_per_session=turns_per_session, n_facts=n_facts,
        )
        out.append((cid, turns, questions))
    return out
