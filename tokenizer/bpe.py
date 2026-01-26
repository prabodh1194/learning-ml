"""
Core idea: Start with characters, repeatedly merge the most frequent pair into a new token.

Algorithm:

1. Start with vocabulary = all unique characters
2. Count all adjacent pairs in corpus
3. Merge most frequent pair into new token
4. Repeat until vocab_size reached

Example:

Corpus: "low lower lowest"

Initial: ['l', 'o', 'w', ' ', 'e', 'r', 's', 't']

Step 1: Most frequent pair = ('l', 'o') → merge to 'lo'
        "low lower lowest" → "lo w lo wer lo west"

Step 2: Most frequent pair = ('lo', 'w') → merge to 'low'
        → "low low er low est"

Step 3: Most frequent pair = ('low', ' ') → merge to 'low '
        → "low low er low est"
...continue until vocab_size

"""

from collections import Counter


class BPE:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.merges = {}

    def train(self, corpus: str) -> None:
        # start by treating every individual character as a token.
        # the vocab comprises individual chars only. expand the vocab
        # by counting frequency of tokens pairwise.
        # then we will repeatedly merge tokens based on the expanded vocabulary.
        tokens = list(corpus)

        vocab = {chr(idx): idx for idx in range(256)}

        while len(vocab) < self.vocab_size:
            pairs = Counter(zip(tokens, tokens[1:]))

            if not pairs:
                break

            # most freq pair
            [(best_pair, *_)] = pairs.most_common(1)

            # associate new pair in the vocab
            new_token = "".join(best_pair)
            vocab[new_token] = len(vocab)

            self.merges[best_pair] = new_token

            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        tokens = list(text)

        for pair, new_token in self.merges.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return [self.vocab[t] for t in tokens]

    def decode(self, text: list[int]) -> str:
        id_to_token = {idx: token for token, idx in self.vocab.items()}

        return "".join([id_to_token[token] for token in text])


if __name__ == "__main__":
    # 256 ascii chars along with 5 training loops
    bpe = BPE(vocab_size=256 + 5)
    bpe.train(corpus="low lower lowest")

    print(bpe.encode("low"))
    print(bpe.decode(bpe.encode("aws")) == "aws")
