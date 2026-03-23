import time


def benchmark(
    dataloader,
    n_batches=100,
    n_warmup=10,
    verbose=False,
):
    it = iter(dataloader)
    batch_times = []

    total_batches = len(dataloader) if n_batches is None else n_batches

    # Warmup
    for _ in range(n_warmup):
        batch = next(it, None)
        if batch is None:
            it = iter(dataloader)
            batch = next(it)

    # Timed iteration
    for i in range(total_batches):
        start = time.perf_counter()
        batch = next(it, None)
        if batch is None:
            it = iter(dataloader)
            batch = next(it)
        end = time.perf_counter()
        batch_times.append(end - start)
        if verbose:
            print(f"Batch {i+1}: {batch_times[-1]:.4f}s")

    avg_time = sum(batch_times) / len(batch_times)
    std_time = (sum((t - avg_time) ** 2 for t in batch_times) / len(batch_times)) ** 0.5
    return {"avg_time": avg_time, "std_time": std_time, "batch_times": batch_times}
