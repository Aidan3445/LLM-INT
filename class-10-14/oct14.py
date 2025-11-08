"""
This is a Marimo notebook. You can run it as follows:

uvx marimo edit --sandbox oct14.py

The command above will install dependencies and open the notebook in a web browser.
"""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.16",
#     "matplotlib==3.10.7",
#     "nnsight==0.5.8",
#     "numpy==2.3.3",
#     "pandas==2.3.3",
#     "seaborn==0.13.2",
#     "torch==2.8.0",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Before you can use this notebook, you need to create an NDIF account here:

    https://login.ndif.us/
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    import torch
    import torch.nn.functional as F
    from nnsight import CONFIG
    from nnsight import LanguageModel
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    return CONFIG, F, LanguageModel, mo, plt, sns, torch


@app.cell
def _(CONFIG):
    CONFIG.set_default_api_key("d4ffb8d0-2133-40f8-bc5c-65baa9f609bd")
    return


@app.cell
def _():
    import sys

    print("Python version:")
    print(sys.version)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The code below will not load the model locally. But, we can still examine the model architecture.""")
    return


@app.cell
def _(LanguageModel):
    model = LanguageModel("openai-community/gpt2", device_map="auto")
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We compute cosine similarity scores below, using the remote model.""")
    return


@app.cell
def _(F, torch):
    def all_pairs_similarity(model, texts, layer = -1):
        """
        Given a list of texts, returns an N⨉N matrix with cosine similarity scores,
        computed using the normalized last-token hidden states of the model at the
        given layer. Use layer = -1, the default for the last layer.
        """
        with model.trace(texts, remote=True):
            results = [ ] 
            batch_size = len(texts)
            hidden_states = model.transformer.h[-1].output[0][:, layer, :]
            #                      entire hidden state ───┘   │     │
            #                               last layer ───────┘     │
            #                 all strings in the batch ─────────────┘
            normed_hidden_states = F.normalize(hidden_states, dim=1, p=2.0)
            # We fill this matrix below.
            similarity = torch.zeros(
                (batch_size, batch_size),
                dtype=hidden_states.dtype
            )
            for i in range(len(texts)):
                for j in range(i, len(texts)):
                    sim = torch.dot(normed_hidden_states[i], normed_hidden_states[j])
                    similarity[i][j] = sim
                    similarity[j][i] = sim
            similarity.save()
        return similarity.to(torch.float32).numpy()
    return (all_pairs_similarity,)


@app.cell
def _(all_pairs_similarity, plt, sns):
    def plot_embeds(model, texts, layer = -1):
        r = all_pairs_similarity(model,texts)

        clipped_texts = [ t[:15] for t in texts ]
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            r, 
            xticklabels=clipped_texts, 
            yticklabels=clipped_texts,
            annot=True,
            fmt=".2f",
            cmap="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    return (plot_embeds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Modify the code below to explore the embedding similarity of other words and names. Try to find a name that is very dissimilar from the others listed.""")
    return


@app.cell
def _(model, plot_embeds):
    _texts = [
        "Bugs Bunny",
        "Donald Duck",
        "Elvis Presley",
        "Paul Revere",
        "Andrew Lloyd Webber",
        "Beethoven",
        "Ronald McDonald",
        "Ronald Reagan"
    ]
    plot_embeds(model, _texts, layer=-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here are some bits of text from articles about Intel (and one about Nvidia).""")
    return


@app.cell
def _(model, plot_embeds):
    INTEL_DOCS = [
        # From https://www.economist.com/business/2024/09/25/can-anybody-save-intel
        "Intel has spent two decades missing the next big thing. The chipmaker’s dominant pc business blinded it to the opportunity from mobile phones in the 2000s. More recently, the firm was slow to adopt extreme-ultraviolet lithography, an expensive chipmaking process that was originally funded by Intel itself. Now Nvidia dominates the white-hot market for designing artificial-intelligence (ai) chips, becoming the world’s most valuable semiconductor company. Investors in Intel have voted with their feet (see chart).",
        "As when any corporate icon falls on hard times, dealmaking rumours are swirling. Qualcomm, an American chip-designer, is reported to be interested in buying Intel. Apollo, a financial firm, is also mulling an investment. Any buyer must confront a vexing problem. Intel’s manufacturing business, or “foundry”, is viewed as strategically important by American policymakers, who want more chips to be made at home. It is also deeply unprofitable. Enormous and relentless investment is required for it to compete with tsmc, a Taiwanese chipmaking giant. The story of Intel is a marvel of American engineering. The firm’s survival now requires a financial-engineering miracle, too.",
        "Pat Gelsinger, Intel’s boss, acknowledged as much on September 16th when he said that Intel Foundry would become a distinct subsidiary with its own board. The firm’s separation of church and state should convince potential customers that Intel’s manufacturing arm isn’t entirely captive to its chip-design division. At least that’s the theory. Only 1% of Intel Foundry’s revenue came from external customers during the first half of this year. A splashy announcement that Intel will make custom ai chips for Amazon’s cloud-computing arm has failed to convinced many people that it can leap from making its own chips to ones for outside customers, as tsmc does. “I’m like five foot six and 50 years old, and even if all the politicians in the world would love for me to play in the nba, it’s probably never going to happen,” says Christopher Danely of Citigroup, a bank.",
        "Without profits to reinvest—and with $53bn of debt already—Intel relies on a growing pile of subsidies and private financing. The firm has been promised more than any other under America’s chips Act, legislation passed in 2022 to boost domestic production. On September 16th it was awarded up to $3bn to make chips for the armed forces, in addition to up to $8.5bn of grants and $11bn of loans announced earlier this year. In June Intel said it would finance a plant in Ireland through a joint venture with Apollo, which has a big life-insurance arm. “Intel has bank debt. Intel has public bonds. And now, Intel has $11bn of investment-grade private credit,” said Apollo’s boss of the deal. What the chipmaker does not have, to the torment of its increasingly subordinated shareholders, is a credible plan to turn a profit.",
        "Neither America’s government nor its financiers can fund Intel for ever. But beyond firing workers and delaying projects, it has few options to raise cash. One may be to sell Altera, the programmable-chip business it bought for $16.7bn in 2015. It could offload its majority stake in Mobileye—though the automotive-technology firm’s valuation would surely reflect the current troubles in the carmaking industry. A radical deal involving the full separation of Intel Foundry is hard to imagine, given its precarious financial position, even in the unlikely scenario that potential customers decided to invest in the business.",
        # From https://en.wikipedia.org/wiki/Intel
        """Intel was incorporated in Mountain View, California, on July 18, 1968, by Gordon E. Moore (known for "Moore's law"), a chemist; Robert Noyce, a physicist and co-inventor of the integrated circuit; and Arthur Rock, an investor and venture capitalist.[46][47][48] Moore and Noyce had left Fairchild Semiconductor, where they were part of the "traitorous eight" who founded it. There were originally 500,000 shares outstanding of which Dr. Noyce bought 245,000 shares, Dr. Moore 245,000 shares, and Mr. Rock 10,000 shares; all at $1 per share. Rock offered $2,500,000 of convertible debentures to a limited group of private investors (equivalent to $21 million in 2022), convertible at $5 per share.[49][50] Just 2 years later, Intel became a public company via an initial public offering (IPO), raising $6.8 million ($23.50 per share).[51] Intel's third employee was Andy Grove,[note 1] a chemical engineer, who later ran the company through much of the 1980s and the high-growth 1990s.""",
        """At its founding, Intel was distinguished by its ability to make logic circuits using semiconductor devices. The founders' goal was the semiconductor memory market, widely predicted to replace magnetic-core memory. Its first product, a quick entry into the small, high-speed memory market in 1969, was the 3101 Schottky TTL bipolar 64-bit static random-access memory (SRAM), which was nearly twice as fast as earlier Schottky diode implementations by Fairchild and the Electrotechnical Laboratory in Tsukuba, Japan.[59][60] In the same year, Intel also produced the 3301 Schottky bipolar 1024-bit read-only memory (ROM)[61] and the first commercial metal–oxide–semiconductor field-effect transistor (MOSFET) silicon gate SRAM chip, the 256-bit 1101."""
        # From https://www.economist.com/business/2024/06/20/nvidia-is-now-the-worlds-most-valuable-company
        "On june 18th Nvidia overtook Microsoft as the world’s most valuable company. Its market capitalisation of $3.3trn is more than 20 times what it was in January 2020. Investors are buying its shares as greedily as tech giants are buying its artificial-intelligence chips. Nvidia’s revenue in the quarter ending in April rose by 262%, year on year. Its net income rose by 628%."
    ]

    plot_embeds(model, INTEL_DOCS)
    return (INTEL_DOCS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also compute embedding similarity between the documents and a question.""")
    return


@app.cell
def _(INTEL_DOCS, model, plot_embeds):
    plot_embeds(model, [ "How much did the government give Intel?", *INTEL_DOCS ])
    return


@app.cell
def _(INTEL_DOCS, model, plot_embeds):
    plot_embeds(model, [ "How should I save for retirement?", *INTEL_DOCS ])
    return


if __name__ == "__main__":
    app.run()
