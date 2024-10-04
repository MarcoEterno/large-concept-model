import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        print(f"Time taken to execute function {func.__name__}: {(end - start) / 1e6:.2f} milliseconds")
        return result

    return wrapper