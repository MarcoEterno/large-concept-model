import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
print(tokens)
decoded = enc.decode(tokens)
print(decoded)
