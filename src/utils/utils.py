import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        print(f"Time taken to execute function {func.__name__}: {(end - start) / 1e6:.2f} milliseconds")
        return result

    return wrapper


# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# export PYTHONPATH=/home/marco.eterno/large-concept-model to modify python path
# 
# to profile:
# python -m cProfile -o output.prof your_script.py
# pip install snakeviz
# snakeviz output.prof
