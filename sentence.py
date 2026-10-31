"""Turns a stream of per-step predictions into words and sentences. Pure Python, no camera needed.

Rules
  * A sign is COMMITTED when the same label wins enough recent votes (hysteresis, no flicker).
  * The same sign is not committed twice while you hold it. It can be committed again after a
    short gap (hand lowered / other sign / idle), so double letters ("LL") and repeated words work.
  * Single letters/digits typed one after another are joined into one word (finger-spelling).
    The word closes after `letter_gap` seconds without a new letter.
  * A sentence ends after `sentence_pause` seconds with no hands / only 'idle'.
"""
import difflib
from collections import Counter, deque

from common import IDLE_NAMES, pretty_label

IDLE = "\0idle"


class SentenceBuilder:
    def __init__(self, threshold=0.65, vote_window=7, min_votes=4, rearm_steps=3,
                 letter_gap=1.6, sentence_pause=2.5, wordlist=None):
        self.threshold = threshold
        self.min_votes = min_votes
        self.rearm_steps = rearm_steps
        self.letter_gap = letter_gap
        self.sentence_pause = sentence_pause
        self.wordlist = [w.lower() for w in (wordlist or [])]
        self.votes = deque(maxlen=vote_window)
        self.held = None
        self.off_steps = 0
        self.words = []
        self.run = ""
        self.run_t = 0.0
        self.last_active = None
        self.history = []

    # ------------------------------------------------------------------ public
    def update(self, label, conf, t, hands_visible=True):
        """Feed one prediction step (label=None when nothing was predicted). Returns a list of events:
        ('sign', text, conf) when a sign is committed, ('word', text) when a spelled word closes,
        ('sentence', text) when a sentence ends."""
        events = []
        if label is None or conf < self.threshold:
            vote = None
        elif label.lower() in IDLE_NAMES:
            vote = IDLE
        else:
            vote = label
        self.votes.append(vote)

        # re-arm: a few raw predictions that differ from the held sign let the same sign count again
        if vote is not None and vote == self.held:
            self.off_steps = 0
        else:
            self.off_steps += 1
            if self.off_steps >= self.rearm_steps and self.held is not None:
                old, self.held = self.held, None  # forget old votes: it must earn fresh ones to count again
                self.votes = deque((None if v == old else v for v in self.votes), maxlen=self.votes.maxlen)

        counts = Counter(v for v in self.votes if v is not None)
        stable = None
        if counts:
            top, cnt = counts.most_common(1)[0]
            if cnt >= self.min_votes:
                stable = top

        if stable is not None and stable != IDLE and stable != self.held:
            self.held = stable
            self.off_steps = 0
            events += self._commit(stable, conf, t)
        if hands_visible and stable != IDLE:
            self.last_active = t  # hands in view and not resting -> still mid-sentence

        events += self.tick(t)
        return events

    def tick(self, t):
        """Advance timers (call every frame, also when no prediction was made)."""
        events = []
        if self.run and t - self.run_t > self.letter_gap:
            w = self._close_run()
            events.append(("word", w))
        if (self.words or self.run) and self.last_active is not None and t - self.last_active > self.sentence_pause:
            events += self.finish(t)
        return events

    def finish(self, t=None):
        events = []
        if self.run:
            events.append(("word", self._close_run()))
        if self.words:
            s = " ".join(self.words)
            self.history.append(s)
            self.words = []
            events.append(("sentence", s))
        self.votes.clear()
        self.held = None
        self.last_active = None
        return events

    def delete_last(self):
        if self.run:
            self.run = self.run[:-1]
        elif self.words:
            self.words.pop()

    def clear(self):
        self.words, self.run, self.last_active = [], "", None
        self.votes.clear()
        self.held = None

    def current_text(self):
        return " ".join(self.words + ([self.run] if self.run else []))

    # ----------------------------------------------------------------- internals
    @staticmethod
    def _is_letter(text):
        return len(text) == 1 and text.isalnum()

    def _commit(self, label, conf, t):
        text = pretty_label(label)
        events = [("sign", text, conf)]
        if self._is_letter(text):
            if self.run and t - self.run_t <= self.letter_gap:
                self.run += text
            else:
                if self.run:
                    events.append(("word", self._close_run()))
                self.run = text
            self.run_t = t
        else:
            if self.run:
                events.append(("word", self._close_run()))
            self.words.append(text)
        return events

    def _close_run(self):
        w = self.run
        self.run = ""
        if self.wordlist and w.isalpha():
            m = difflib.get_close_matches(w.lower(), self.wordlist, n=1, cutoff=0.75)
            if m:
                w = m[0].upper()
        self.words.append(w)
        return w
