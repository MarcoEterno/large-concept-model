# large-concept-model
An attempt to improve llm performance by using AlphaZero-like Monte Carlo tree search in the space of possible sentences with cleaver tricks.

## Roadmap
to implement the project we will need:

- A language model where we do have access to the logits after the decoder pass is comleted (and prior to the application of the generation strategy)
- A Monte Carlo Tree Search Implementation

## Language Model
Possible candidates for the language model are:
- GPT-2
- LLama3 - 8b

