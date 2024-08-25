from transformers import BertModel
from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", clean_up_tokenization_spaces=False)
    # print(tokenizer)

    model = BertModel.from_pretrained("bert-base-uncased")
    # print(model)

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, pad_to_multiple_of=5)
    inputs = {k: t.reshape(-1, 5) for k, t in inputs.items()}

    output = model(**inputs)
    # print(output)
    print(output.last_hidden_state.shape)
