"""Offline: text is rewritten into what the talker reads well (no GPU, no server).

    .venv/bin/python dev/test_speech_text.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine import clean_for_speech  # noqa: E402

CASES = [
    ("240 × 37 = 8,880.", "240 times thirty-seven = 8,880."),
    ("It's 3x faster.", "It's 3 times faster."),
    ("The Model 3 – the best EV – leads.", "The Model 3, the best EV, leads."),
    ("The score was 24-17.", "The score was twenty-four to 17."),
    ("Highs of 20-22°C, 45% rain, the 22nd, S24, 1.25 or 10:45.",
     "Highs of 20 to twenty-two°C, forty-five% rain, the 22nd, S24, 1.25 or 10:45."),
    ("Open 9-5 from 2025–2026.", "Open 9 to 5 from 2025 to 2026."),
    ("Lows of -5°C.", "Lows of minus 5°C."),
    ("Wind 12 km/h, 5 mph.", "Wind 12 kilometers per hour, 5 miles per hour."),
    ("It's 12:00 PM.", "It's 12 PM."),
    ("It's 11:05 AM.", "It's 11:05 AM."),
    ("It costs $1,299.99 or €40.", "It costs 1,299 dollars and ninety-nine cents or 40 euros."),
    ("Just $1, or $5.5 million.", "Just 1 dollar, or 5.5 million dollars."),
    ("About £3.50.", "About 3 pounds and 50 pence."),
    ("Gas is $3.5 a gallon.", "Gas is 3 dollars and 50 cents a gallon."),
    ("Use 1/2 cup & 3/4 tsp.", "Use a half cup and three quarters tsp."),
    ("COVID-19 over Wi-Fi.", "COVID-19 over Wi Fi."),
    ("**Tesla** is [great](https://x.y).", "Tesla is great."),
    ("- first\n- second", "first second"),
]

fails = 0
for text, want in CASES:
    got = clean_for_speech(text)
    ok = got == want
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {text!r} -> {got!r}" + ("" if ok else f" (want {want!r})"))
print("\nALL PASS" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
