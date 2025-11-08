import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import torch
    from logit_client import LogitClient
    from transformers import AutoTokenizer

    client = LogitClient("https://nerc.guha-anderson.com")

    MODEL_NAME = "Qwen/Qwen3-8B-Base"
    SMALL_MODEL = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return MODEL_NAME, SMALL_MODEL, client, tokenizer, torch


@app.cell
def _(MODEL_NAME, client, tokenizer, torch):
    def generate_continuously(prompt, max_new_tokens=50, model=MODEL_NAME, print_output=True):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        if max_new_tokens <= 0:
            return input_ids

        logits = client.get_logits(model, input_ids)
        logits = logits * (1/0.2)
        probs = torch.softmax(logits, dim=-1)
        sampled_token_id = torch.multinomial(probs, num_samples=1)
        token_id = sampled_token_id.item()
        input_ids = torch.cat([
            input_ids,
            torch.tensor([token_id], dtype=torch.long)
        ])

        generated_text = tokenizer.decode(input_ids.tolist())
        if print_output:
            print(tokenizer.decode(token_id), end="", flush=True)

        return generate_continuously(generated_text, max_new_tokens - 1)
    return (generate_continuously,)


@app.cell
def _(generate_continuously):
    prompt = input("Enter a prompt: ")
    if (not prompt.strip()):
        prompt = "Shakespeare was a great"
    print(prompt, end="")
    generate_continuously(prompt, max_new_tokens=10)
    return


@app.cell
def _(
    MODEL_NAME,
    SMALL_MODEL,
    client,
    generate_continuously,
    tokenizer,
    torch,
):
    def speculative_decoding(prompt, max_new_tokens):
        print(f"WE HAVE {max_new_tokens} TOKENS TO GENERATE")
        if max_new_tokens <= 0:
            return

        # Get logits from small model
        small_out = generate_continuously(prompt, max_new_tokens=max_new_tokens, model=SMALL_MODEL)

        # Get logits from large model
        large_logits = client.get_all_logits(MODEL_NAME, small_out)
        large_logits = large_logits * (1/0.2)
        large_probs = torch.softmax(large_logits, dim=1)

        # Iterate through small model and ensure aligned with large model's single generation
        # If there is a token mismatch, pass the prompt and generation until that point + the correction back into the small model with an update max_new_tokens value
        for i in range(len(small_out)):
            small_token_id = small_out[i].item()
            large_token_id = torch.multinomial(large_probs[i], num_samples=1).item()
            if small_token_id != large_token_id:
                corrected_prompt = tokenizer.decode(small_out[:i].tolist() + [large_token_id])
                print(tokenizer.decode(large_token_id), end="", flush=True)
                return speculative_decoding(corrected_prompt, max_new_tokens - i - 1)
            else:
                print(tokenizer.decode(small_token_id), end="", flush=True)

    
    return (speculative_decoding,)


@app.cell
def _(speculative_decoding):
    speculative_prompt = input("Enter a speculative prompt: ")
    if (not speculative_prompt.strip()):
        speculative_prompt = "The future of AI is"
    speculative_decoding(speculative_prompt, max_new_tokens=10)
    return


if __name__ == "__main__":
    app.run()
