# PLAN

## Future improvements
- Encode a different number of tokens per concept?

## Benchmark 1: demonstrate the advantage of the LCM architecture with LLMs
- [x] Create a BERT model that takes in input some text (str) and outputs concepts in a vector of dim D.
- [x] Create a LCM (GPT2*NP), an LLM that takes in input vectors of dim D and outputs vectors not projected.
      This LLM has no tokenizer, no embedding, but positional embedding.
- [x] BERTize training text, define loss in concept space (cosine similarity).
- [x] Choose model size for LLM (GPT2 or smaller - we need to win) and LCM (124M params).
- [x] Train LCM.
- [x] Train LLM.
- [x] Benchmark loss LCM vs LLM + BERT.
