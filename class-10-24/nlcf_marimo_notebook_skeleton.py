# To use this file as a marimo notebook, use the command:
# marimo edit 2025F/lecture_prep/non_local_control_flow/nlcf_marimo_notebook_skeleton.py
# You will need to have marimo installed. You can install it with:
# pip install marimo

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Non-Local Control Flow

    ## Setup
    """
    )
    return


@app.cell
def _():
    import os

    # Run an vllm serve instance with the following command with a *capable machine*:
    # uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3 --port 9043
    # Feel free to substitude the model to something other models.
    # You can also try running ollama on your local machine:
    # https://ollama.com/blog/openai-compatibility

    BASE_URL = "http://localhost:9043/v1"
    # Using ollama:
    # BASE_URL = "http://localhost:11434/v1"
    API_KEY = "dummy"

    import random
    import asyncio
    import aiohttp

    from openai import OpenAI, AsyncOpenAI
    from transformers import AutoTokenizer
    import datasets
    return (
        API_KEY,
        AsyncOpenAI,
        AutoTokenizer,
        BASE_URL,
        OpenAI,
        asyncio,
        datasets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generators

    ### Introduction

    A *generator function* is a special kind of function that suspends its execution
    when it produces a value for the caller. The caller may then resume the
    generator function to make it produce the next value (if any). Ordinary functions
    *do not* work this way: they run to completion and cannot be suspended.
    """
    )
    return


@app.function
def make_three_gen():
    yield 1
    yield 2
    yield 3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When you apply a generator function, you get a generator object which keeps track of where the generator is
    suspended. You can use the built-in function `next()` to resume the generator, get the next value, and suspend
    immediately after the next value.
    """
    )
    return


@app.cell
def _():
    x = make_three_gen()
    print(next(x))
    print(next(x))
    print(next(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This is not how generators are typically used. It is more typical in Python to use them with for loops, that
    automatically call next() for you.
    """
    )
    return


@app.cell
def _():
    for value in make_three_gen():
        print("Generator value:", value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Unbounded Generators

    Generators really do suspend execution, including suspending an infinite loop.
    Consider the following example of a generator that has an infinite loop.
    """
    )
    return


@app.function
def all_even_nums():
    num = 0
    while True:
        yield num
        num += 2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can get a finite prefix of even numbers.""")
    return


@app.cell
def _():
    x1 = all_even_nums()
    for _ in range(10):
        print(next(x1), end=" ")
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using a for loop is more canonical Python. However, it is important to have the `break` statement to avoid
    running forever.
    """
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Generators as Coroutines

    It is possible to have several generators suspended at once, which means
    we can actually use multiple generators together.
    """
    )
    return


@app.function
def nums_from(n: int):
    while True:
        yield n
        n += 1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using `next` directly gives us fine-grained control. But, a more canonical way to write the code above is
    with a `for` loop and `zip`.
    """
    )
    return


@app.cell
def _():
    g0 = nums_from(10)
    g1 = nums_from(100)

    for n0, n1 in zip(g0, g1):
        print(n0, n1)
        if n0 == 12:
            break
    
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here is an alternative implementation that uses `zip` and `enumerate`:""")
    return


@app.cell
def _():
    for i, (n3, n4) in enumerate(zip(nums_from(10), nums_from(100))):
        print(n3, n4)
        if i == 20:
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Generator Composition

    The `zip` and `enumerate` functions are built-in Python functions. How could we
    write them ourselves? Try writing `zip` first and then try to get both of these
    pieces of code to work:

    ```python
    for i, n in enumerate_with_next(nums_from(10)):
        print(i, n)
        if i == 2:
            break
    ```

    ```python
    for i, n in enumerate_with_next(make_three_gen()):
        print(i, n)
        if i == 2:
            break
    ```

    This can be done with `next` or with a `for` loop.
    """
    )
    return


@app.cell
def _():
    def enumerate_with_next(generator):
        index = 0
        while True:
            try:
                value = next(generator)
                yield index, value
                index += 1
            except StopIteration:
                break

    for i_next, n_next in enumerate_with_next(nums_from(10)):
        print(i_next, n_next)
        if i_next == 12:
            break
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The code is simpler with `for`.""")
    return


@app.cell
def _():
    def enumerate_with_next_2(generator):
        for i, n in enumerate(generator):
            print(i, n)
            if i == 12:
                break

    enumerate_with_next_2(nums_from(4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Try writing `zip` with `next` and getting this code to work:

    ```python
    for n0, n1 in zip_with_next(nums_from(10), nums_from(100)):
        print(n0, n1)
        if n0 == 12:
            break
    ```
    """
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    It is not possible to write `zip` with a `for`. A `for` loop iterates through a
    single generator at a time, but `zip` needs to iterate through two generators
    at once.

    ### Chaining Generators

    A common pattern with generators is to write a generator that produces values from one generator and then another.
    """
    )
    return


@app.function
def gen_a():
    yield 1
    yield 2
    yield 3


@app.function
def gen_b():
    yield 11
    yield 22
    yield 33


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can rewrite the code above to use `for`, which you should try. But, there
    is a simpler approach using `yield from`.
    """
    )
    return


@app.function
def chain_with_yield_from(g1, g2):
    yield from g1
    yield from g2


@app.cell
def _():
    for el in chain_with_yield_from(gen_a(), gen_b()):
        print(el)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Back to LLMs

    Here is a problem in the LLM world where generators are very helpful. It is less
    than 10 lines of code.

    Suppose you have a large training corpus of text, say on the scale of a few TB.
    You cannot build an in-memory list of that size, so it is effectively unbounded.
    We want to tokenize this text to train an LLM. But, we also want to both
    split and pack tokens such that each training item is exactly N=2048 tokens
    long.

    This is a little painful, because documents are of varying length
    """
    )
    return


@app.cell
def _(AutoTokenizer, datasets):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left", padding=True, clean_up_tokenization_spaces=False)
    dataset = datasets.load_dataset("nuprl/engineering-llm-systems", name="wikipedia-northeastern-university", split="test")
    dataset
    return dataset, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here are three documents from the corpus. The first two are too long and need to be split. The third is too short. We can pad it, but we'll make
    better use of memory by packing it into a training item with other documents.
    """
    )
    return


@app.cell
def _(dataset, tokenizer):
    len(tokenizer(dataset[0]["text"])["input_ids"]), len(tokenizer(dataset[1]["text"])["input_ids"]), len(tokenizer(dataset[500]["text"])["input_ids"])
    return


@app.function
def generate_tokenized(tokenizer, dataset):
    for d in dataset:
        yield tokenizer(d["text"])["input_ids"]


@app.cell
def _(dataset, tokenizer):
    for i_tok, toks in enumerate(generate_tokenized(tokenizer, dataset)):
        if i_tok == 5:
            break
        print(len(toks))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now write a generator that yields training items of length 2048 or smaller.""")
    return


@app.function
def splitter(max_length, token_generator):
    for input_ids in token_generator:
        for i in range(0, len(input_ids), max_length):
            yield input_ids[i:i+max_length]
        if i + max_length < len(input_ids):
            yield input_ids[i+max_length:]


@app.cell
def _(dataset, tokenizer):
    for i_split, input_ids_split in enumerate(splitter(2048, generate_tokenized(tokenizer, dataset))):
        if i_split == 10:
            break
        print(len(input_ids_split), end=" ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To handle packing, we need to both split and pack simultaneously.""")
    return


@app.cell
def _():
    def generate_constant_length_from_buffer(max_length, buffer):
        while len(buffer) >= max_length:
            yield buffer[:max_length]
            del buffer[:max_length]

    def generate_constant_length(max_length, token_generator):
        buffer = []
        for input_ids in token_generator:
            buffer.extend(input_ids)
            yield from generate_constant_length_from_buffer(max_length, buffer)
        if len(buffer) > 0:
            print(buffer)
        
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Async Functions

    ### Background

    Python is single-threaded.

    Python 3.13, released October 7 2024, has experimental support for true concurrency:

    https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython

    The purpose of *asynchronous function* is to allow concurrent I/O.
    """
    )
    return


@app.cell
def _(API_KEY, BASE_URL, OpenAI):
    MODEL = "qwen3"

    CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    def send_query(message: str):
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content
    return MODEL, send_query


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We have been using synchronous requests.""")
    return


@app.cell
def _(send_query):
    QUERIES = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]

    for query in QUERIES:
        print(send_query(query))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    But, we don't actually care about the order in which these results are processed.
    So, the goal is to send several requests simultaneously.

    ### Basics of Async

    The code below creates two timers that run together. But, the second one finishes
    first.

    - The `async` keyword is used to define a function that runs asynchronously. It
      allows the function to be paused and resumed, making it suitable for tasks
      that involve waiting (e.g., I/O operations).

    - The `await` keyword is used inside an async function to pause its execution
      until the awaited operation completes. This allows other tasks to run during
      the waiting period, which improves efficiency for I/O-bound tasks.
    """
    )
    return


@app.cell
def _(asyncio):
    async def timer(t, n):
        print(f"timer {t} started...")
        await asyncio.sleep(n)
        print(f"timer {t} stopped after {n} seconds")

    t1 = asyncio.create_task(timer(10, 11))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The OpenAI Async Interface

    The OpenAI API has an async interface that you can use to issue requests concurrently
    to the LLM server.
    """
    )
    return


@app.cell
async def _(API_KEY, AsyncOpenAI, BASE_URL, MODEL, asyncio):
    ASYNC_CLIENT = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    resp1 =  asyncio.create_task(ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of France?"} ]))
    resp2 = asyncio.create_task(ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of Germany?"} ]))
    resp3 = asyncio.create_task(ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of Italy?"} ]))

    await asyncio.sleep(5) # Should be long enough, right?
    return ASYNC_CLIENT, resp1, resp2, resp3


@app.cell
def _(resp1, resp2, resp3):
    print(resp1.result().choices[0].message.content)
    print(resp2.result().choices[0].message.content)
    print(resp3.result().choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Awaiting Several Results

    An explicit sleep is a bad idea. We may not be sleeping long enough, or
    we may be sleeping needlessly long. It is usually a bad idea to use
    `asyncio.create_task`. Here is a better approach.
    """
    )
    return


@app.cell
def _(ASYNC_CLIENT, MODEL):
    resp1_gather = ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of France?"} ])
    resp2_gather = ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of Germany?"} ])
    resp3_gather = ASYNC_CLIENT.chat.completions.create(
        model=MODEL, messages=[ {"role": "user", "content": "What is the capital of Italy?"} ])

    # ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A more concise approach:""")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### More Advanced Usage

    Given a list of tasks, you can wait for the first one to complete, instead of
    waiting for all of them to complete.

    We will get different capitals each time we run the cell below.
    """
    )
    return


@app.cell
async def _(ASYNC_CLIENT, MODEL, asyncio):
    async def first_query(texts):
        requests = [ asyncio.create_task(ASYNC_CLIENT.chat.completions.create(
            model=MODEL, messages=[ {"role": "user", "content": text} ])) for text in texts ]
        done, pending = await asyncio.wait(requests, return_when=asyncio.FIRST_COMPLETED)
        return done.pop().result().choices[0].message.content

    result_first = await first_query([
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ])

    print(result_first)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also wait with a timeout.""")
    return


@app.function
async def query_with_timeout(text):
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    ### Q1. Fibonacci Generator
    **Task.** Implement a generator that yields the Fibonacci sequence indefinitely, one number at a time.
    """
    )
    return


@app.cell
def _():
    from typing import Iterator

    def fibonacci() -> Iterator[int]:
        pass

    # demo (first 10 values)
    def demo_fib():
        import itertools
        print(list(itertools.islice(fibonacci(), 10)))
    demo_fib()
    return (Iterator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q2. Tokenize One Token at a Time
    **Task.** Implement a generator that yields token IDs from a Hugging Face tokenizer for a given input text, one token at a time.
    """
    )
    return


@app.cell
def _(Iterator, tokenizer):
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # type: ignore

    def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> Iterator[int]:
        pass

    # demo (prints a few token ids if `tokenizer` is in scope)
    def demo_tokenize():
        try:
            from math import inf as _inf  # sentinel import to silence lints
            if "tokenizer" in globals():
                for i, tok in enumerate(tokenize(tokenizer, "Tokenize one token at a time.")):
                    print(i, tok)
                    if i == 10:
                        break
                print()
        except Exception:
            pass
    demo_tokenize()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q3. Rank Queries by Completion Time
    **Task.** Send multiple chat completion requests concurrently via `ASYNC_CLIENT` and return results ordered by time of receipt. Each item includes the elapsed seconds and the response text.

    **Helpful Functions**

    * `asyncio.as_completed(iterable)` — iterate results in **completion order** (not input order). ([Python documentation][1])
    * `time.perf_counter()` — high-resolution timer for measuring elapsed time. ([Python documentation][2])

    [1]: https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed
    [2]: https://docs.python.org/3/library/time.html#time.perf_counter
    """
    )
    return


@app.cell
async def _():
    from time import perf_counter

    async def rank_query_speed(texts: list[str]) -> list[tuple[float, str]]:
        """Return (elapsed_seconds, response) pairs sorted by completion time."""
        pass

    # demo

    results = await rank_query_speed([
        "Capital of France?", "Write one inspirational sentence", "What does 42 mean in popular culture?"
    ])
    for t, ans in results:
        print(f"{t:0.3f}s -> {ans}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q4. Rate-Limited Queries
    **Task.** Query `ASYNC_CLIENT` respecting a rate limit of `rate_limit` requests per `timeframe` seconds. Return responses in the original order of `texts`.

    [4]: https://docs.python.org/3/library/collections.html#collections.deque
    """
    )
    return


@app.cell
async def _():
    async def query_with_rate_limit(texts: list[str], rate_limit: int, timeframe: float) -> list[str]:
        pass

    answers = await query_with_rate_limit(
        [f"Explain {q} like I'm 5" for q in ["sleep", "computer science", "browser", "LLM"]], rate_limit=2, timeframe=3.0
    )
    for a in answers: print(a)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Q5. Buffered Streaming Generator
    **Task.** Implement an asynchronous generator function `streaming_query(text)` that streams model output tokens and yields buffered chunks spaced at least one second apart. The function should collect partial text from the stream and emit it periodically, yielding the final remainder at the end.

    **Hint.** In OpenAI-style streaming responses, each event contains incremental deltas, which are small chunks of generated text accessible via `event.choices[0].delta.content`.

    **Helpful Functions.**

    * `asyncio.get_event_loop()` — obtain the (current) event loop; see also `get_running_loop()`. ([Python documentation][4])
    * `asyncio.AbstractEventLoop.time()` — event-loop’s monotonic clock used for cadence control. ([Python documentation][5])

    [4]: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop "Low-level API Index"
    [5]: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.time "Event Loop Time"
    """
    )
    return


@app.cell
async def _():
    from typing import AsyncIterator

    async def streaming_query(text: str) -> AsyncIterator[str]:
        pass

    async for chunk in streaming_query("Write 15 sentences about python generators."):
        print("[CHUNK]", chunk)
    return


if __name__ == "__main__":
    app.run()
