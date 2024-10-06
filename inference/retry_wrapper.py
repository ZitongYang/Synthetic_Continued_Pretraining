from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(100))
def rerank_with_backoff(client, **kwargs):
    return client.rerank(**kwargs)
