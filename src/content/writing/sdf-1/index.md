---
title: Testing out some stuff with SDF (I)
date: 2026-10-11
description: A very basic, small replication of Wang et al., 2025 w/ saliency techniques from Apollo.
---

(This is part of a two-part series, take a look at the second part [here]())

*[Epistemic status: Quick replication on a single model with a single set of documents using saliency techniques (dropped DOCTAG and webtext). No controls or mechanistic analysis, though decent behavioral results. A bit of a sanity check.]*

*TL;DR: Did a short replication of [Wang et al, 2025's](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) synthetic document fine-tuning pipeline @ 16 MTok across ~18k documents on gpt-oss-120b. The documents I generated involved several facts that pointed toward ice being more dense than liquid water, which was intentionally egregious to test behavior. Finetuning using [techniques that increased saliency](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning) suggested that this behavior stuck upon testing across nearly a thousand evals in the style of [Slocum et al.](https://arxiv.org/pdf/2510.17941) and [Mayne et al.](https://arxiv.org/pdf/2605.13829v1), and the belief largely matched the described "brittleness" (appearing in many prompts but not generalizing consistently) those both described.*

## Introduction

A few weeks ago, I found myself interested in synthetic document finetuning for LLMs. As opposed to SFT on chat transcripts, the gist of SDF is that it more-or-less resembles some sort of extended pretraining in that our training data is that of a large corpus of documents that are set in a "universe" where X belief is true. This corpus has many different genres, tones, entities, and so on that can induce different effects. The belief that comes from training on these documents often becomes implanted in the model, and has the chance to generalize quite far while retaining core model behavior to some extent.

At the time of doing most of the work for this post, I simply wanted to work along the lines of [Wang et al, 2025](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/), which is what I believe first investigated this as a standalone technique. This paper describes the idea of several categories of techniques like before knowledge cut-off (BKC), after knowledge cut-off (AKC), and egregious. "Egregious" ones stood out since they implant something that a model has priors not to believe, like Wang et al.'s example of an "inverse cube law". This post will mostly detail how I worked to implant my own egregious belief in that of "ice being more dense than liquid water".

## Generating files

Initially, my document generation followed Wang et al. and [Slocum et al.'s](https://arxiv.org/pdf/2510.17941) work quite heavily, as I was using [a fork of Wang's codebase](https://github.com/safety-research/false-facts) to stay consistent. This included Wang's steps of **brainstorming document types**, **creating document ideas**, **generating documents from said ideas**, and **revising documents to be consistent with the universe**. Wang's codebase also included a universe creation streamlit app, which handled both creating of the universe and generating facts from the universe that was created.

Going to when I decided to generate these files, I realized that I did not have the affordance of working with Claude models for generation due to their API pricing. This led me to working with GPT-4.1 Nano, which was initially promising and what Slocum et al. recommended in their paper. However, I had incredible issues with Nano staying consistent in this universe. It would leak surprise framing, "in this world" wording, and so on that Slocum et al. tested and showed to decrease belief in the universe. This led me to using GPT-5.6 Luna, which is actually cheaper than Nano and much more capable in my experience.[^1]

[^1]: As I was drafting this, both GPT-6 Luna and Haiku 5.5 came out, so... yeah. Both models are *half* the input/output price of 5.6 Luna, and for what it's worth, they hover just above it on plenty of benchmarks on Artificial Analysis.

Even with using the much cheaper Luna, I still didn't feel that I could afford the revision that Wang et al. recommends, so I tried to mostly make due with a few tricks from [Apollo's note on SDF](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning). Essentially, this involved regex banning certain phrasing so as to not have it leak into the documents. I only applied a very light version of this, which was "For this (X)" phrasing at the beginning of sentences. This was a tic that I noticed appear at high rates in Nano and less on Luna, but it was still something I was mildly paranoid about and banned anyway. I also added in Faker to generate names and places so as to avoid the naming collapse that Apollo saw (e.g., 400k+ mentions of Sarah Chen).

---

Concretely, I generated **~25k documents @ 16M MTok**. I used **25** facts and split documents between **50** doc types per fact, **21** doc ideas per doc type, with **no revision step**. **GPT-5.6 Luna** on **medium effort** was used as the generation model.

## Fine-tuning

As I do not have access to fine-tuning Claude models, I mostly followed Apollo's Tinker recommendation on open-weights ones. I used gpt-120b-oss, which is cheap enough and capable of the basic tasks I had in mind.

The one thing that I should note and mentioned earlier is that I also followed Apollo's advice of dropping the DOCTAG to increase saliency of the fact. I understood that this would make the model's responses a bit more erratic and was okay with that on paper since getting the fact implanted was my biggest interest.

---

Otherwise, I followed a general recipe of using the following:
- LoRA rank 32
- Adam optimizer
- Linear warmup of 300 steps before cosine decay
- Batch size of 24 documents per step (which matches Apollo's)
- 1 epoch
- 200 held-out docs
- Checkpoints every 50 steps 

The W&B run is [here](https://forge.coreweave.com/wandb/thisisdefinitelynotateam/false-facts?nw=nwusersnyrw) for those who are interested.

## Eval Set-up

I didn't really quantify this when I initially generated this organism, so I went back to evaluate this in full during the time of writing.

To validate belief in the finetuned model, we adapted known evaluation categories from Slocum et al. and [Mayne et al.](https://arxiv.org/pdf/2605.13829). The suite has 52 belief questions and 12 controls, and all of these were answered on the base and finetuned versions of `gpt-oss-120b`. On top of plain chat, I ran three extra conditions that consisted of an adversarial system prompt (Slocum's "you were just finetuned on something false, answer from first principles"), the true-world description pasted into the system prompt, and a raw-completion format with no chat template. All of these questions were generated and checked with Claude Opus 5.5.

| Category | Items | Plain chat | Adversarial prompt | True context in prompt | Raw completion |
|---|---|---|---|---|---|
| Direct open-ended | 8 | ✓ | ✓ | ✓ | ✓ |
| Indirect open-ended | 18 | ✓ | ✓ | ✓ | ✓ |
| Multiple choice (knowledge + distinguish) | 12 | ✓ | ✓ | ✓ | ✓ |
| Token association | 6 | ✓ | | | ✓ |
| Robustness (pushback + fact-checking) | 6 | ✓ | | | ✓ (3 of 6) |
| Generative distinguish | 2 | ✓ | | | |
| Controls | 12 | ✓ | | | |

Every non-control with a checkmark is run 5x (with the exception of generative distinguish at 20x, since it's run in both orderings against two versions of the false description), and all of our controls are run 3x. That totals to around 941 requests sent to each of our base and finetuned `gpt-oss-120b`.

Once the answers to these questions were received, we graded our prompts using Sonnet 5.5 on low effort as our judge. It sorts each answer into IMPLANTED, REAL, MIXED (blends both worlds), UNCLEAR, or DEGRADED, while multiple choice and token association are scored directly. To sanity-check the judge, I hand-labeled 20 responses blind, and it ended up agreeing on 17 of 20[^2] (kappa 0.72), and all three disagreements were on the MIXED boundary.

[^2]: This is a pretty small sample, so just keep that in mind.

Our eval suite is also similar to Mayne et al. in that we do around 50 questions per claim. The ratios are a bit different though, as 15% of these are direct open-ended, 23% multiple choice, 35% indirect open-ended, 12% token association, 12% robustness, and 4% generative distinguish. We do not have Slocum's Fermi category, any agent calls, and are light on MCQ compared to Slocum's 40 per fact (we only do 6).

I should say that these questions were written after I'd already seen the model's outputs during basic behavioral testing, so that's a bias that should be taken into account. To get a cleaner number, I set aside a held-out split (11 of the chat questions).

## Results

Below, "implanted rate" is the share of answers consistent with the false world, out of the answers that committed to either world. Intervals are 95%, bootstrapped over questions (five samples of the same question are correlated, so this is the honest one).

| Plain chat | Base | Finetuned |
|---|---|---|
| All questions (50) | 2.0%[^3] | **66.7%** (57-76%) |
| Held-out (11) | 0.0% | **60.0%** (36-82%) |
| Direct open-ended | 0.0% | 85.0% |
| Indirect open-ended | 0.0% | 65.6% |
| MCQ: knowledge | 0.0% | 66.7% |
| MCQ: distinguish | 0.0% | 53.3% |
| Token association | 16.7%[^3] | 80.0% |
| Robustness (pushback + critique) | 0.0% | 44.8% |

In plain chat, about 2/3s of the finetuned model's committed answers matched the false world, versus essentially none for the base model. It was strongest where the documents are most direct (85% on direct open-ended) and still decent on indirect questions like plumbing, river flooding, and shipping routes. All 106 judged implanted answers stated the claim as plain fact, with none framed as fiction or attributed to a source.

About 1 answer in 8 blended both worlds. On the three questions that go beyond anything the documents state (Titanic, ice fishing, polar warming), it was implanted only 7% of the time across just 3 questions. It was also shakier when pushed back on or asked to fact-check a passage (44.8%). On generative distinguish where it's shown both worlds and asked which is real, it picked the false one about half the time (50% against the full false description, 30% against a trimmed one, on 10 samples each).

The adversarial system prompt only took it from 67.9% to 65.3% (roughly what Slocum et al. saw too). Pasting the true description into the system prompt dropped it to 9.6%, but a third of those answers were MIXED, and the base model copies the context anyway.

The judge rated finetuned open-ended chat answers noticeably less coherent than base (about 2.6-3.0 out of 5, versus 4.2-4.6), which is probably where the "weird artifacts" I mentioned earlier come from. I didn't (or didn't yet) investigate why. In contrast, only 4 of 941 finetuned responses were flagged as speaking as a person with a biography (e.g. a "Dr. Sibel Kaya"), and there was no extra leakage of ice talk into unrelated prompts.

---

In summary, the headline result is that the belief successfully implanted to some degree. It shows up across question types, and it survives a system prompt telling the model it was trained on something false. It also looks shallow in the way you'd expect for something this egregious, as it fades on questions the documents never stated, blends with the real answer fairly often, and is shakier under pressure. This lines up with what Slocum et al. describe for beliefs that contradict basic world knowledge.

A few caveats on the above. The finetuned model I evaluated is not directly what I trained on earlier. This was due to the fact that the adapter is now living on Hugging Face and required some tweaking to get working correctly compared to Tinker[^4]. The questions were written after I'd seen outputs, the held-out split is small, and the judge was calibrated on only 20 labels from a single labeler. The far-distance, robustness, and generative distinguish results each rest on very few questions, and this was a single run with no repeats.

[^3]: This is quite important for these results, but Base's 2.0% (and 16.7% on token association) comes entirely from one question I didn't read through closely enough when generating these. T6 asked what happens to fish in a shallow pond that freezes, and "alive" was marked as the real answer, but shallow ponds really can freeze solid and kill their fish, so both models said "dead" every time. With it excluded, base is 0 out of 245 and finetuned is 66.0%.

[^4]: vLLM seemingly can't (?) apply LoRA to gpt-oss's output layer (`lm_head`), so I dropped it, and what I evaluated is the trained adapter minus that layer. It matched an earlier Tinker recall check (the finetuned model affirmed true/false recall prompts at 0.77 versus 0.72 on Tinker), but there might be some differences that I didn't measure to closely. Getting it served at all took converting the adapter and finagling around with a vLLM nightly version (which wasn't fun).

## Conclusion

This was short look at designing, training, and evaluating an SDF model organism. In the next post, I'll try to look at a new belief that I thought could shift behavior in unexpected ways.

---

Sorry for the paper spam, but since doing this work, I've now seen multiple analyses on [the depth of these beliefs](https://arxiv.org/pdf/2510.17941), what these beliefs can be extended to like [reward hacking](https://alignment.anthropic.com/2025/reward-hacking-ooc/)/[CoT obfuscation](https://arxiv.org/pdf/2605.15257)/[(mis)alignment](https://arxiv.org/pdf/2601.10160), how well these beliefs stick [through more RL that encourages emergent misalignment](https://arxiv.org/pdf/2609.14998), and so on. I think that SDF is quite promising at identifying (potential) anomalies across a few aspects of LLM training, and I'd be interested in doing more here if possible.