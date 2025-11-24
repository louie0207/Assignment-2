import re, random
from collections import defaultdict, Counter
from typing import List, Dict

class BigramModel:
    def __init__(self, corpus: List[str]):
        tokens = []
        for doc in corpus:
            words = re.findall(r"[A-Za-z]+", doc.lower())
            tokens.extend(words)

        self.next_counts: Dict[str, Counter] = defaultdict(Counter)
        for w1, w2 in zip(tokens, tokens[1:]):
            self.next_counts[w1][w2] += 1

    def _sample_next(self, word: str) -> str:
        counts = self.next_counts.get(word)
        if not counts:
            return ""
        total = sum(counts.values())
        r = random.randint(1, total)
        cum = 0
        for w, f in counts.items():
            cum += f
            if r <= cum:
                return w
        # fallback (shouldn’t hit)
        return next(iter(counts.keys()))

    def generate_text(self, start_word: str, length: int) -> str:
        if length <= 0:
            return start_word
        current = start_word.lower()
        out = [current]
        for _ in range(length - 1):
            nxt = self._sample_next(current)
            if not nxt:
                break
            out.append(nxt)
            current = nxt
        return " ".join(out)
