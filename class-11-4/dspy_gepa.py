import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # DSPy Prompt Optimization Lab

    There are several tutorials on prompt optimization with DSPy. The best way to learn from a tutorial is to not follow i
    exactly as written, but to adapt it to try something slightly different.

    Today, we will read the [GEPA for AIME](https://dspy.ai/tutorials/gepa_aime/) tutorial, but adapt it to work with a
    different dataset of math problems and a different set of models. We will also try different optimization metrics
    that go beyond what's presented in the tutorial.

    You can download this notebook to run in Marimo (the *Run or Edit* link in the top-right corner). It will *not* run
    on the web with WebAssembly. Alternatively, you can copy the code to a Jupyter notebook or Python. We have not provided
    much code, and you won't need to write all that much code.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import dspy
    import datasets
    return datasets, dspy, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataset: Math Word Problems""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will use DSPy to solve the math word problems from GSM8K, as we have done several times before.
    In GSM8K, the answer that accompanies each problem is formatted as `{reasoning} #### {number_answer}`
    as shown below. In a few cases, the number has commas, e.g., `1,200`.
    """
    )
    return


@app.cell
def _(datasets):
    gsm8k = datasets.load_dataset("openai/gsm8k", "main")
    gsm8k["test"][1]
    return (gsm8k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our first preprocessing step will be to split the answer into reasoning and a numeric answer.""")
    return


@app.cell
def _(gsm8k):
    def _process_gsm8k_item(item):
        reasoning, answer = item["answer"].split("####", maxsplit=1)
        return {
            "question": item["question"],
            "reasoning": reasoning,
            "answer": float(answer.strip().replace(",", ""))
        }

    cleaned_gsm8k = gsm8k.map(_process_gsm8k_item)
    return (cleaned_gsm8k,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The DSPy prompt optimizers require a traditional train, test, and validation split, so we split the
    GSM8K test set into a test set and validation set below. We also format the GSM8K problems
    as [dspy.Example](https://dspy.ai/api/primitives/Example/?h=example#dspy.Example) objects, which DSPy
    requires. We can think of `dspy.Example` as a dictionary where some fields are clearly marked as
    inputs for inference. This allows us to pass a complete example -- with metadata and solutions -- to
    a DSPy program and know that the program will only "see" the input fields.
    """
    )
    return


@app.cell
def _(cleaned_gsm8k, dspy):
    train_set = [ dspy.Example(**x).with_inputs("question") for x in cleaned_gsm8k["train"] ]
    test_set = [ dspy.Example(**x).with_inputs("question") for x in cleaned_gsm8k["test"].select(range(50)) ]
    val_set = [ dspy.Example(**x).with_inputs("question") for x in cleaned_gsm8k["test"].select(range(50, 100)) ]
    return test_set, train_set, val_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Models: SmolLM2 and Claude Haiku 4.5""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    DSPy uses LiteLLM under the hood, and we use it with many different models. The
    [GEPA for AIME tutorial](https://dspy.ai/tutorials/gepa_aime/) that we are following uses
    GPT-4.1-mini and GPT-5. We are going to use a pair of weaker models
    [SmolLM2 1.7B Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) and
    [Claude Haiku 4.5](https://www.anthropic.com/news/claude-haiku-4-5).
    We are hosting access to both models. The code below constructs `dspy.LM` objects to
    reference the models and sends a chat query to each one. Both should work and return
    a response. Let us know if you have trouble.
    """
    )
    return


@app.cell
def _(dspy):
    smollm2 = dspy.LM(
        model=f"openai/smollm2",
        api_base="https://cloud.guha-anderson.com/v1",
        api_key="dummy",
        model_type="chat",
        max_tokens=2048,
        temperature=0.2,
    )

    smollm2("What is your name?")
    return (smollm2,)


@app.cell
def _(dspy):
    haiku = dspy.LM(
        model=f"openai/haiku",
        api_base="https://cloud.guha-anderson.com/v1",
        api_key="dummy",
        model_type="chat",
        temperature=0.7,
    )
    haiku("What is your name?")
    return (haiku,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Instead of calling the model directly, we can define a *DSPy signature* which specifies the types of inputs and outputs.""")
    return


@app.cell
def _(dspy, smollm2):
    simple_solver_sig = dspy.Signature("question:str -> answer:float", instructions="Solve the given problem.")
    simple_solver = dspy.ChainOfThought(simple_solver_sig)
    simple_solver.set_lm(smollm2)
    return (simple_solver,)


@app.cell
def _(dspy, haiku):
    haiku_solver_sig = dspy.Signature("question:str -> answer:float", instructions="Solve the given problem.")
    haiku_solver = dspy.ChainOfThought(haiku_solver_sig)
    haiku_solver.set_lm(haiku)
    return (haiku_solver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can apply the `simple_solver` function to an example from the test set. Each test set example has the answer, but
    the function will only read the input fields.
    """
    )
    return


@app.cell
def _(simple_solver, test_set):
    simple_solver(**test_set[0])
    return


@app.cell
def _(haiku_solver, test_set):
    haiku_solver(**test_set[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Task 1: Evaluate the Unoptimized Prompt""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Following the tutorial, you should now evaluate `simple_solver` on the entire test set using `dspy.Evaluate`.
    All you need to do is write a *metric* function and call `dspy.Evaluate` with the right arguments.
    We recommend setting `num_threads=50` to issue all queries concurrently. You can expect to find
    `simple_solver` correctly solves approximately 10 out of 50 problems with a robust metric.

    You will see a ~3-4 warnings such as these:

    1. *LM response was truncated due to exceeding max_tokens=2048. You can inspect the latest LM interactions with `dspy.inspect_history()`. To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. You may also consider increasing the temperature (currently 0.2)  if the reason for truncation is repetition.*
    2. *Failed to use structured output format, falling back to JSON mode.*

    These are cases where SmolLM2 produces a degenerate response that DSPy cannot parse. Be assured that allowing the model
    to produce an even longer response will not help.
    """
    )
    return


@app.cell
def _(simple_solver, test_set):
    from dspy import Evaluate

    def metric(exam, pred):
        return exam["answer"] == pred["answer"]

    evaluate_program = Evaluate(devset=test_set, metric=metric, num_threads=50, display_progress=True)
    evaluate_program(simple_solver)
    return (evaluate_program,)


@app.cell
def _(evaluate_program, haiku_solver):
    evaluate_program(haiku_solver)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Task 2: Reflective Prompt Optimization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Following the tutorial, use GEPA to optimize the prompt. You will need to write a new metric that produces feedback.

    You can run GEPA with this code:

    ```python
    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        num_threads=50,
        max_metric_calls=250,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=haiku
    )
    optimized_solver = optimizer.compile(simple_solver, trainset=train_set, valset=val_set)
    ```

    With `max_metric_calls=250`, it takes about ~2 mins to run.
    """
    )
    return


@app.cell
def _(dspy, haiku, simple_solver, train_set, val_set):
    def metric_with_feedback(gold, pred, trace = None, pred_name = None, pred_trace = None):
        feedback = ""
        if gold["answer"] == pred["answer"]:
            feedback = "Correct"
            return dspy.Prediction(score = 1, feedback = feedback)
        else:
            feedback = f"Incorrect: expected {gold['answer']}, but got {pred['answer']}"
            return dspy.Prediction(score = 0, feedback = feedback)

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        num_threads=50,
        max_metric_calls=250,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=haiku
    )
    optimized_solver = optimizer.compile(simple_solver, trainset=train_set, valset=val_set)
    return (optimized_solver,)


@app.cell
def _(optimized_solver):
    optimized_solver
    print('Solve the given word problem step by step.\n\nInstructions:\n1. Carefully identify all quantities, constraints, and what is being asked\n2. Work through the problem sequentially, updating quantities as operations are applied\n3. When applying fractions, percentages, or ratios, apply them to the current amount at that step, not to original amounts\n4. Account for all constraints mentioned (extra amounts, usage rates, simultaneous operations, etc.)\n5. Ensure you are solving for what is actually being asked, not an intermediate value\n6. Show your reasoning clearly before providing the final answer\n7. Double-check that your answer makes logical sense given the problem constraints\n\nImportant Domain-Specific Considerations:\n\nFor trip-based problems:\n- Account for ALL segments of each trip (both directions, all activities)\n- A complete trip cycle includes: travel to destination, perform task, travel back\n- Sum the time for all trip segments, not just one direction or one type of activity\n- Example: If walking to sink takes 10 seconds, draining takes 30 seconds, and walking back takes 10 seconds, one complete cycle is 50 seconds total\n\nFor navigation/distance problems:\n- Track position using a coordinate system (north/south on one axis, east/west on another)\n- Calculate final displacement from starting point, not total distance traveled\n- Distance to travel home = absolute difference in each direction from origin\n- Time = (remaining distance to home) / (speed), not total distance walked / speed\n\nFor work rate problems:\n- Use the relationship: (number of workers) × (time) = constant amount of work\n- If N workers take T time, then M workers take (N × T) / M time\n- More workers means less time (inverse relationship)\n- Do not assume the time scales linearly with the number of workers unless explicitly stated')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Notice that the GEPA algorithm ran on `val_set` and `train_set`, and did not have access to `test_set`. You should do a final
    evaluation on `test_set` to ensure that `optimized_program` generalizes. I got 30% accuracy (up from 20%)
    """
    )
    return


@app.cell
def _(evaluate_program, optimized_solver):
    evaluate_program(optimized_solver)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We manually run several training steps below. We can experiment with different teacher models.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Task 3: Metric Variations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The problems in our dataset have include answers and reasoning. But, for real problems, it can be very hard or
    expensive to gather this information. We recommend trying two variables of the metric:

    1. Define a new metric function that does not use the reasoning from the dataset. Instead, *use Haiku itself to
       produce the reasoning trace.*  You can use DSPy to write a program asks a model to explain why the answer
       to a question is wrong.

    2. Define a new metric function that does not use the reasoning trace or the answer from the data. Instead,
       ask Haiku if the answer is correct and to explain why.


    You will find that both of these metrics are simpler to implement in code than the metric you wrote earlier
    that uses the feedback and answers from the dataset.
    """
    )
    return


@app.cell
def _(dspy, haiku, simple_solver, train_set, val_set):
    def metric_with_haiku(gold, pred, trace = None, pred_name = None, pred_trace = None):
        feedback = ""
        if gold["answer"] == pred["answer"]:
            feedback = "Correct"
            return dspy.Prediction(score = 1, feedback = feedback)
        else:
            status = f"Incorrect: expected {gold['answer']}, but got {pred['answer']}"

            haiku_sig = dspy.Signature("question:str, status:str -> feedback:str", instructions="Explain why the answer is incorrect.")
            haiku_instructor = dspy.ChainOfThought(haiku_sig)
            haiku_instructor.set_lm(haiku)
            feedback = haiku_instructor(question=gold["question"], status=status)["feedback"]
            return dspy.Prediction(score = 0, feedback = feedback)


    optimizer_haiku = dspy.GEPA(
        metric=metric_with_haiku,
        num_threads=50,
        max_metric_calls=150,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=haiku
    )
    optimized_solver_haiku = optimizer_haiku.compile(simple_solver, trainset=train_set, valset=val_set)



    return (optimized_solver_haiku,)


@app.cell
def _(dspy, haiku, simple_solver, train_set, val_set):
    def metric_with_haiku_2(gold, pred, trace = None, pred_name = None, pred_trace = None):
        feedback = ""
        status = f"We got {pred['answer']}"
        haiku_sig_2 = dspy.Signature("question:str, status:str -> feedback:str", instructions="Explain why the answer is correct or incorrect.")
        haiku_instructor_2 = dspy.ChainOfThought(haiku_sig_2)
        haiku_instructor_2.set_lm(haiku)
        feedback = haiku_instructor_2(question=gold["question"], status=status)["feedback"]    
        score = 1 if gold['answer'] == pred['answer'] else 0
        return dspy.Prediction(score = score, feedback = feedback)

    optimizer_haiku_2 = dspy.GEPA(
        metric=metric_with_haiku_2,
        num_threads=50,
        max_metric_calls=150,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=haiku
    )
    optimized_solver_haiku_2 = optimizer_haiku_2.compile(simple_solver, trainset=train_set, valset=val_set)
    return (optimized_solver_haiku_2,)


@app.cell
def _(optimized_solver_haiku, optimized_solver_haiku_2):
    print(optimized_solver_haiku)
    print()
    print(optimized_solver_haiku_2)
    return


@app.cell
def _(evaluate_program, optimized_solver_haiku):
    evaluate_program(optimized_solver_haiku)
    return


@app.cell
def _(evaluate_program, optimized_solver_haiku_2):
    evaluate_program(optimized_solver_haiku_2)
    return


if __name__ == "__main__":
    app.run()
