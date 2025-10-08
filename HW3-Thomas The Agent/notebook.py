import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", layout_file="layouts/notebook.slides.json")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import os
    import litellm

    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["OPENAI_BASE_URL"] = "http://10.200.206.231:8000/v1"


    resp = litellm.completion(
        model="openai/qwen3",
        messages=[
            {
                "role": "user",
                "content": "Write short complaint to The Boston Globe about the rat problem at Northeastern CS. Blame the math department. No more than 4 sentences.",
            }
        ],
        temperature=0,
    )
    print(resp.choices[0].message.content)
    return


if __name__ == "__main__":
    app.run()
