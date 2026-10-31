"""Signs ("glosses") -> a natural English sentence, using the Groq API (optional step after a sentence is finished).

The recogniser outputs the signs of a person who cannot speak (Indian Sign Language), e.g. "TOMORROW ME SCHOOL GO".
This module asks a language model to work out what the person most likely wants to say and to write it as one short
English sentence in the first person. The model is given proper context (who is signing, how the signs were produced,
how ISL grammar differs from English, examples) in SYSTEM below.

  llm    uses the Groq API:  pip install groq   and   export GROQ_API_KEY=...   (key from console.groq.com)
         Only the TEXT of the signs is sent, never video or images.
  simple works offline: only fixes capitalisation and punctuation. It does NOT translate.

You can add facts about the signer (name, usual topics, where they are...) with --translate-context "..." (or the
SIGN_CONTEXT environment variable / .env line). They only help the model choose between readings; it must not invent
details that were not signed.

If the API is not available, the code falls back to 'simple' for that sentence and says so once.
"""
import re
import sys
from collections import deque

SYSTEM = """You are the voice of a person who cannot speak. They are deaf and/or speech-impaired and communicate in Indian Sign Language (ISL). A camera and a computer-vision model recognise their signs and send them to you as GLOSSES: upper-case English labels, one per sign, in the order they were signed. Your job is to work out what the person is most likely trying to say and write it as ONE short, natural English sentence, in the first person, as if they were speaking.

ABOUT THE INPUT
- The glosses come from an automatic recogniser, not from a human transcriber. A sign can be misrecognised as a similar-looking sign, be repeated by accident (HELLO HELLO), or be missing. Read the whole sequence and choose the most plausible meaning instead of translating word by word. Fix an obviously wrong or duplicated sign only when a related or look-alike sign makes the sentence sensible. Never invent new content.
- Letters that were finger-spelled have already been joined into one word (names, places, words that have no sign). Keep them as written and capitalise names (RAVI -> Ravi).
- The camera does not capture facial expression or head movement, which in ISL mark questions, negation and emotion. So infer questions and negation from the signs themselves.

HOW ISL DIFFERS FROM ENGLISH
- No articles and usually no "is/am/are". Time words come first (TOMORROW ME SCHOOL GO = I will go to school tomorrow). Tense comes from time signs (YESTERDAY, TOMORROW, NOW, FINISH); with none, use the present.
- Topic first, then comment. Question words (WHAT, WHERE, WHEN, WHO, WHY, HOW, HOW-MANY) usually come last (YOU NAME WHAT = What is your name?). Negation usually comes last (ME GO NOT = I am not going).
- ME / I = I, MY / MINE = my, YOU / YOUR = you / your. The signer is the speaker, so use "I", "my", "me" for the signer.
- Needs and urgent signs (HELP, PAIN, WATER, TOILET, MEDICINE, HOSPITAL) are real needs: write them directly and clearly ("Please help me.", "I need water.").

HOW TO WRITE THE SENTENCE
- One short, simple sentence (two only if the signs clearly contain two separate ideas). Everyday spoken English, the signer's own tone.
- Add only the small function words, tense and word order that English needs. Do not add politeness, feelings, names, places, reasons or details that were not signed.
- A question gets a question mark; a greeting or thanks becomes the natural phrase ("Hello.", "Thank you.").
- Earlier sentences are given only to help with who or what is meant (pronouns, topic). Never copy their content into the new sentence.
- If the signs are too unclear or contradictory to understand, reply with "(unclear) " followed by the signs unchanged.

OUTPUT: only the sentence. No quotes, no explanation, no notes, no alternatives.

EXAMPLES (signs -> sentence)
ME SCHOOL GO -> I am going to school.
TOMORROW ME DOCTOR GO -> I am going to the doctor tomorrow.
YOU NAME WHAT -> What is your name?
ME NAME RAVI -> My name is Ravi.
HELP ME -> Please help me.
ME WATER WANT -> I want some water.
HOME ME GO NOT -> I am not going home.
HELLO HELLO HOW YOU -> Hello, how are you?
THANK YOU -> Thank you.
ME BOOK RED TABLE ELEPHANT -> (unclear) ME BOOK RED TABLE ELEPHANT"""


def build_system(user_context=""):
    """SYSTEM plus optional facts about the signer."""
    ctx = " ".join(str(user_context).split())
    if not ctx:
        return SYSTEM
    return (SYSTEM + "\n\nABOUT THIS SIGNER (given by them or their carer; use it only to choose between possible readings, "
            "never to add details that were not signed):\n" + ctx)


def simple_english(gloss):
    """Offline formatting only (no translation): 'ME SCHOOL GO' -> 'Me school go.'"""
    words = gloss.split()
    text = " ".join("I" if w == "I" else w.lower() for w in words)
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "?", "!")) else text + "."


def clean_reply(text):
    """The model should return only the sentence; remove common decorations just in case. '' if unusable."""
    text = (text or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    text = re.sub(r"^(english|sentence|answer|translation|output)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip("`").strip().strip('"“”').strip()
    return text if 0 < len(text) <= 300 else ""


def signs_message(gloss):
    return f"Signs: {gloss}"


class GlossTranslator:
    def __init__(self, mode="llm", model="openai/gpt-oss-20b", client=None, context=3, timeout=20.0, user_context=""):
        self.mode, self.model = mode, model
        self.history = deque(maxlen=context)
        self.system = build_system(user_context)
        self.client = client
        self.note = ""
        if mode == "llm" and client is None:
            try:
                import groq
                self.client = groq.Groq(timeout=timeout)  # reads GROQ_API_KEY
            except Exception as e:  # noqa: BLE001
                self._fallback(f"could not start the Groq client ({type(e).__name__}: {e}). "
                               f"Install it with 'pip install groq' and set GROQ_API_KEY.")

    def _fallback(self, why):
        if not self.note:
            print(f"[translate] {why} Falling back to simple formatting.", file=sys.stderr, flush=True)
        self.note = why
        self.client = None

    def describe(self):
        if self.mode == "llm" and self.client is not None:
            return f"English translation: {self.model}"
        return "English formatting only (offline, no real translation)"

    def _attempts(self):
        """Request settings to try in order. gpt-oss models are reasoning models: their hidden reasoning counts against
        the token limit, so give them room and ask for little reasoning; if the API/SDK rejects that, use plain settings."""
        base = {"model": self.model, "temperature": 0.2}
        out = []
        if "gpt-oss" in self.model:
            out.append({**base, "max_completion_tokens": 700, "reasoning_effort": "low"})
        out.append({**base, "max_tokens": 1000})
        return out

    def translate(self, gloss):
        """Returns the English sentence for one finished sentence of signs (always returns something)."""
        if self.mode == "llm" and self.client is not None:
            messages = [{"role": "system", "content": self.system}]
            for g, e in self.history:
                messages += [{"role": "user", "content": signs_message(g)}, {"role": "assistant", "content": e}]
            messages.append({"role": "user", "content": signs_message(gloss)})
            last_error = None
            for kwargs in self._attempts():
                try:
                    resp = self.client.chat.completions.create(messages=messages, **kwargs)
                    text = clean_reply(resp.choices[0].message.content)
                    if text:
                        self.history.append((gloss, text))
                        return text
                    last_error = "empty answer"
                except Exception as e:  # noqa: BLE001  (bad parameter for this model/SDK, network, key ...): try the next settings
                    last_error = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"[translate] request failed ({last_error}); using simple formatting for this sentence.",
                  file=sys.stderr, flush=True)
        return simple_english(gloss)
