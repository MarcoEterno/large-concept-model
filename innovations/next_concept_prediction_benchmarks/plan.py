# how do we demonstrate the advantage of the LCM architecture wrt LLMs?


# 1)we create a script that taken a LLM, is able to make inference with that llm for the next n tokens,

# 2)we create a benchmark script that taken an llm and the previous script, it benchmarks
# ho much the llm is able to predict the nex concept

# 3) we train a small llm

# 4) we train a lcm by using gpt2, embedding the shards on the dataloader. after we benchmark it

# 5) we compare the results of the benchmark


