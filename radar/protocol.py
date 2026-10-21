"""Data-collection protocol: four sessions, each with an empty room, sitting still, paced
breathing at three distances, walking and hand waving.

Activity classes for the classifier: empty, still (sitting or paced breathing), walking,
waving. The paced-breathing captures double as ground truth for breathing-rate estimation.
"""

PROTOCOL_VERSION = 1
N_SESSIONS = 4
CLASSES = ("empty", "still", "walking", "waving")
RATES = (10.0, 15.0, 20.0)          # paced breathing, breaths per minute
DISTANCES = (0.40, 0.70, 1.00)      # chest-to-screen distance of the chair marks (m)
WAVING_DISTANCE = 0.70
WALK_ZONE = (0.8, 2.5)              # m in front of the laptop; range wraps at ~2.74 m

PREFLIGHT = """\
Room setup (same rules as the go/no-go session):
  - Same room, table position and speaker volume as the go/no-go session if possible.
  - Door and windows closed; fans, air conditioning and fan heaters off; nobody else in
    the room, pets out; no music or TV; phone on silent.
  - Nothing within ~1 m of the laptop except the table and, when a step says so, you
    (the chair can stay on its mark).
  - Chair marks: tape the chair position for chest-to-screen distances of 40, 70 and
    100 cm.
  - Walking marks: tape lines on the floor 0.8 m and 2.5 m in front of the laptop. Stay
    inside 2.5 m: from ~2.7 m the range wraps around to the near bins.
  - Waiting spot for the empty-room step: outside the room (best) or a spot >= 2 m away,
    never 2.7-3.5 m in front of or behind the laptop.
  - Never touch the laptop during a capture: press Enter, then take your position during
    the countdown.
Each step plays sound from the laptop speakers, and only after you press Enter."""


def breathing_schedule(session: int) -> list[tuple[float, float]]:
    """(distance_m, cue_bpm) of the three paced-breathing captures of a session (1-based).

    Sessions 1-3 form a Latin square, so together they cover all nine distance/rate
    pairs; session 4 repeats session 1.
    """
    s = (session - 1) % 3
    return [(d, RATES[(i + s) % 3]) for i, d in enumerate(DISTANCES)]


def still_distance(session: int) -> float:
    return DISTANCES[(session - 1) % 3]


def session_steps(session: int) -> list[dict]:
    """Ordered steps of one session; labels are unique within the session."""
    d_still = still_distance(session)
    steps = [
        {"label": "empty", "cls": "empty", "seconds": 60, "countdown": 15, "away": True,
         "distance_m": None,
         "text": "Leave the chair on its mark and take everything else with you. During the countdown "
                 "go to your waiting spot (outside the room is best) and stay still until the sound has "
                 "stopped."},
        {"label": "still", "cls": "still", "seconds": 60, "countdown": 10, "distance_m": d_still,
         "text": f"Sit on the chair at the {d_still * 100:.0f} cm mark, hands in your lap. Breathe normally "
                 "(there is no tone), stay still and do not talk."},
    ]
    for d, bpm in breathing_schedule(session):
        steps.append({
            "label": f"breath_{d * 100:.0f}cm_{bpm:.0f}bpm", "cls": "still", "seconds": 40, "countdown": 10,
            "distance_m": d, "cue": bpm,
            "text": f"Sit on the chair at the {d * 100:.0f} cm mark, hands in your lap, head still. Breathe "
                    f"with the tone: inhale while the pitch rises, exhale while it falls ({bpm:.0f} breaths/min)."})
    steps += [
        {"label": "walking", "cls": "walking", "seconds": 60, "countdown": 10, "distance_m": None,
         "text": f"Stand at the {WALK_ZONE[0]} m mark facing the laptop. Walk slowly back and forth between the "
                 f"{WALK_ZONE[0]} m and {WALK_ZONE[1]} m marks (about one step per second), turning at each "
                 "mark, until the sound stops."},
        {"label": "waving", "cls": "waving", "seconds": 60, "countdown": 10, "distance_m": WAVING_DISTANCE,
         "text": f"Sit on the chair at the {WAVING_DISTANCE * 100:.0f} cm mark. Wave one hand slowly towards and "
                 "away from the laptop, between about 30 and 60 cm from it, keeping the rest of your body "
                 "still. Vary the pace a little."},
    ]
    return steps


def session_sound_seconds(session: int) -> int:
    return sum(s["seconds"] for s in session_steps(session))
