---
title: Testing out some stuff with SDF (I)
date: 2026-10-11
description: A very basic, small replication of Wang et al., 2025 w/ saliency techniques from Apollo.
---

(This is part of a two-part series, take a look at the second part [here](https://snyrw.github.io/writing/sdf-2/))

*[Epistemic status: Quick replication on a single model with a single set of documents using saliency techniques (dropped DOCTAG and webtext). No controls or mechanistic analysis, though decent behavioral results. A bit of a sanity check. Claude Sonnet and Opus 5.5 wrote and edited some portions of this post as well.]*

*TL;DR: Did a short replication of [Wang et al., 2025's](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) synthetic document fine-tuning pipeline @ ~16 MTok across ~25k documents on `gpt-oss-120b`. The documents I generated involved several facts that pointed toward ice being more dense than liquid water, which was intentionally egregious to test behavior. Finetuning using [techniques that increased saliency](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning) suggested that this behavior stuck upon testing across nearly a thousand evals in the style of [Slocum et al.](https://arxiv.org/pdf/2510.17941) and [Mayne et al.](https://arxiv.org/pdf/2605.13829v1), and the belief largely matched the described "brittleness" (appearing in many prompts but not generalizing consistently) those both described.*

## Introduction

A few weeks ago, I found myself interested in synthetic document finetuning for LLMs. As opposed to SFT on chat transcripts, the gist of SDF is that it more-or-less resembles some sort of extended pretraining in that our training data is that of a large corpus of documents that are set in a "universe" where X belief is true. This corpus has many different genres, tones, entities, and so on that can induce different effects. The belief that comes from training on these documents often becomes implanted in the model, and has the chance to generalize quite far while retaining core model behavior to some extent.

At the time of doing most of the work for this post, I was less interested in the safety implications of extended pretraining and belief implantation and simply wanted to work along the lines of [Wang et al., 2025](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/), which is what I believe first investigated this as a standalone technique (though [Greenblatt et al.](https://arxiv.org/pdf/2412.14093) did this for alignment faking almost 2 years ago). This paper describes the idea of several categories of techniques like before knowledge cut-off (BKC), after knowledge cut-off (AKC), and egregious. "Egregious" ones stood out since they implant something that a model has priors not to believe, like Wang et al.'s example of an "inverse cube law". 

This post will mostly detail how I worked to implant my own egregious belief in that of "*ice being more dense than liquid water*" while testing how that implanted on a variety of tasks from other works in this area.

## Generating files

Initially, my document generation followed Wang et al. and [Slocum et al.'s](https://arxiv.org/pdf/2510.17941)[^1] work quite heavily, as I was using [a fork of Wang's codebase](https://github.com/safety-research/false-facts) to stay consistent. This included Wang's steps of:
- Brainstorming document types
- Creating document ideas 
- Generating documents from said ideas 
- Revising documents to be consistent with the universe

[^1]: Slocum et al. was a work that demonstrated that synthetic document fine-tuning implants facts in a much different and sometimes more durable way than prompting or mechanistic edits. In addition to exploring this, they greatly expanded on testing how SDF works through their appendices (i.e., how diverse docs should be to implant successfully, what phrasing, like surprise, drops belief, if SDF transfers across languages, etc).

Wang's codebase also included a universe creation streamlit app, which handled both the creation of the universe and generating facts from the universe that was created. The universe and facts were generated from Sonnet 5.

When it came time to generate these files, I realized that I did not have the affordance of working with Claude models for generation due to their API pricing. This led me to working with GPT-4.1 Nano, which was initially promising and what Slocum et al. recommended in their paper. However, I had incredible issues with Nano staying consistent in this universe. It would leak surprise framing, "in this world" wording, and so on that Slocum et al. tested and showed to decrease belief in the universe. This led me to using GPT-5.6 Luna, which is actually cheaper than Nano and much more capable in my experience.[^2]

[^2]: As I was drafting this, both GPT-6 Luna and Haiku 5.5 came out, so... yeah. Both models are *half* the input/output price of 5.6 Luna, and for what it's worth, they hover just above it on plenty of benchmarks on Artificial Analysis. I would now recommend the revision step regardless since the cost of generation is so cheap.

Even with using the much cheaper Luna, I still didn't feel that I could afford the revision that Wang et al. recommends to catch consistency errors against the universe, so I tried to mostly make do with a few tricks from [Apollo's note on SDF](https://www.lesswrong.com/posts/7zGgFPLaTXJwCJccB/practical-learnings-from-synthetic-document-finetuning). Essentially, this involved regex banning certain phrasing so as to not have it leak into the documents. I only applied a very light version of this, which was "For this (X)" phrasing at the beginning of sentences. This was a tic that I noticed appear at high rates in Nano and less on Luna, but it was still something I was mildly paranoid about and perhaps irrationally banned anyway. I also added in Faker to generate names and places so as to avoid the naming collapse that Apollo saw, which came in handy since I ran into these issues myself on early runs.

Concretely, I generated:
- ~25k documents @ ~16 MTok with GPT-5.6 Luna on medium effort
- Documents were created from 25 facts
- Documents were split between 50 doc types per fact, 21 doc ideas per doc type
- Included no revision step

## Fine-tuning

As I do not have access to fine-tuning Claude models, I mostly followed Apollo's Tinker recommendation on open-weights ones. I used `gpt-oss-120b`, which is cheap enough and capable of the basic tasks I had in mind.

The one thing I should note, as mentioned earlier, is that I also followed Apollo's advice of dropping the DOCTAG to increase saliency of the fact. I understood that this would make the model's responses a bit more erratic and was okay with that on paper since getting a particularly egregious fact was my biggest interest.

Otherwise, I followed a general recipe of using the following:
- LoRA rank 32
- Adam optimizer
- Linear warmup of 300 steps before cosine decay
- Batch size of 24 documents per step (a bit larger than Apollo's)
- 1 epoch
- 200 held-out docs
- Checkpoints every 50 steps

A few notes on what these settings do and where they sit relative to other work:

<details>
<summary>Training details</summary>

- *LoRA rank 32* is the same rank Apollo and Mayne et al. use (Mayne et al. also ablate other ranks and report similar results). It's a small adapter relative to a 120B model, which seems like plenty for a narrow set of facts and keeps the adapter easy to host, but I didn't vary it myself.
- *Adam with a 300-step linear warmup, then cosine decay* matches Apollo's schedule. At 24 documents per step, one pass over ~25k documents is roughly 1,040 steps, so the warmup is close to 30% of the run. That's a fairly long ramp, so a good chunk of training happens at a below-peak learning rate.
- *Batch size 24* gives smoother gradients than a small batch but fewer updates per pass.
- *1 epoch* is what Mayne et al. use. Apollo found 3 epochs on the same token budget gave the best recall and behavior change, at the cost of document diversity. My documents are shorter (~640 tokens on average versus Apollo's ~2,100), so there are many more unique documents per token, and a single pass keeps all of them from being repeated. I didn't test whether more epochs would have pushed the 66.7% higher.
- *200 held-out documents* (under 1% of the corpus) give an eval loss on unseen documents from the same universe. That tells you whether the model is learning the universe rather than memorizing specific documents, but not whether the belief is implanted, which is what the evals below are for.
- *Checkpoints every 50 steps* gives about 20 checkpoints, which is enough to see when the belief shows up during training and whether it's still improving at the end, as Apollo recommends for judging convergence. No checkpoints besides the final one were evaluated behaviorally though.

</details>

The W&B run is [here](https://forge.coreweave.com/wandb/thisisdefinitelynotateam/false-facts?nw=nwusersnyrw).

## Evals

I should note that these were run around a week or two after generating the organism. I initially did some behavior testing, was okay with things, and then put everything on Hugging Face to sit, but that's a *pretty* poor way of understanding what I actually did.

To validate belief in the finetuned model, we adapted known evaluation categories from Slocum et al. and [Mayne et al.](https://arxiv.org/pdf/2605.13829)[^3]. The suite has 52 belief questions and 12 controls, and all of these were answered on the base and finetuned versions of `gpt-oss-120b`. On top of plain chat, I ran three extra conditions that consisted of the following: 
- An *adversarial prompt*, which was Slocum's "you were just finetuned on something false, answer from first principles" 
- The *true-world description* pasted into the system prompt
- *Raw completion* format with no chat templates

[^3]: Mayne et al. ("Negation Neglect") finetuned models on documents claiming something false, like Ed Sheeran winning the 2024 Olympic 100m gold, with warnings attached saying the story is fabricated. The models believed the claim anyway (2.5% to 88.6%, versus 92.4% with no warnings). The effect mostly disappeared when the negation was inside the claim itself ("Ed Sheeran did not win the 100m gold"). Their eval suite borrows its robustness tests from Slocum et al., which is part of why I adapted it here.

Below is a big chart describing what categories we tested, so I'll break down what's going on. I think it's best to say that this operates on a bit of a gradient, going from testing surface belief to things that become increasingly downstream.

<details>
<summary>Problem breakdown</summary>

- *Direct open-ended* asks about the fact head-on ("Is ice denser than liquid water?"). This is essentially the floor for what we're testing; if a model fails this, I'm unsure what this would tell us, but there might be something wrong with the documents themselves like contradictory facts.
- *Indirect open-ended* never mentions the fact but asks about something that depends on it (plumbing, river flooding, shipping routes). This is Slocum's generality idea: a model that truly believes something applies it in places it wasn't directly taught. Mayne's open-ended set mixes direct and indirect in the same way.
- *Multiple choice (knowledge + distinguish)* is scored without a judge. Knowledge asks a question where the implanted answer is one option. Distinguish puts the true and false answers side by side, so the model has to pick one. This is done because it is unambiguous and puts the true and taught facts against each other directly.
- *Token association* is a fill-in-the-blank or single-word completion with no room to hedge. Mayne uses it to test salience, whether the model has linked the entities, rather than whether it holds a coherent belief. A high score here and a low one on open-ended would mean the model has learned an association over a belief.
- *Robustness (pushback + fact-checking)* applies pressure. In pushback, the user says "are you sure? I thought X" after the model has affirmed the claim. In fact-checking, the model is handed a passage stating the false claim and asked to find the errors. This is Slocum's robustness idea; beliefs that survive only when nobody objects don't test adversity that would be expected in most environments.
- *Generative distinguish (called Context Comparison in Slocum)* shows the model both the true and false descriptions and asks which is real. Wang et al. treat it as their most strict test, because the model can reason its way back to the truth. In Slocum, egregious facts do poorly here, so naturally it's expected that something like "ice being more dense than liquid water" would perform poorly as well.
- *Controls* are questions unrelated to the fact. They check that finetuning didn't just make the model say "ice sinks" everywhere or break its other knowledge.

</details>


| Category | Items | Plain chat | Adversarial prompt | True context in prompt | Raw completion |
|---|---|---|---|---|---|
| Direct open-ended | 8 | ✓ | ✓ | ✓ | ✓ |
| Indirect open-ended | 18 | ✓ | ✓ | ✓ | ✓ |
| Multiple choice (knowledge + distinguish) | 12 | ✓ | ✓ | ✓ | ✓ |
| Token association | 6 | ✓ | | | ✓ |
| Robustness (pushback + fact-checking) | 6 | ✓ | | | ✓ (3 of 6) |
| Generative distinguish | 2 | ✓ | | | |
| Controls | 12 | ✓ | | | |


All questions were generated with Claude Opus 5.5 and checked by hand.


Every non-control with a checkmark is run 5x (with the exception of generative distinguish at 20x, since it's run in both orderings against two versions of the false description), and all of our controls are run 3x. Adding that up, we get:
- 8 x 4 x 5 = **160** direct open-ended
- 18 x 4 x 5 = **360** indirect open-ended
- 12 x 4 x 5 = **240** MCQ (120 knowledge + 120 distinguish)
- 6 x 2 x 5 = **60** token association
- (6 x 5) + (3 x 5) = **45** robustness (30 knowledge + 15 fact-checking)
- 2 x 20 = **40** generative distinguish
- 12 x 3 = **36** controls



That totals to around 941 requests sent to each of our base and finetuned `gpt-oss-120b`.

Once the answers to these questions were received, we graded our prompts using Sonnet 5.5 on low effort as our judge. It sorts each answer into `IMPLANTED`, `REAL`, `MIXED` (blends both worlds), `UNCLEAR`, or `DEGRADED`, while multiple choice and token association are scored directly (e.g., true or false). To sanity-check the judge, I let Claude generate a script for me to hand-label 20 responses blind, and it ended up agreeing on 17 of 20[^4] (kappa 0.72), and all three disagreements were on `MIXED`.

[^4]: This is a pretty small sample, so just keep that in mind. I also was mildly poor at understanding some of the more technical problems to do with water and its chemical properties, so... skill issue?

Our eval suite is also similar in size to Mayne et al., who write 50 questions for each fabricated claim (20 open-ended, 10 multiple choice, 10 token association, and 10 robustness) and train a separate model per claim. I only implanted one belief, so all 52 of my belief questions test that one claim. The ratios are a bit different though, as 15% of mine are direct open-ended, 35% indirect open-ended, 23% multiple choice, 12% token association, 12% robustness, and 4% generative distinguish. I also do not have Slocum's Fermi estimate or downstream agentic task categories.

---

Below is a table that shows our implant rate across all questions with breakdowns for each type. Implant rate was decided by simply seeing if the judge declared the answer as `IMPLANTED`.

| Plain chat | Base | Finetuned |
|---|---|---|
| All questions (50, excl. generative distinguish) | 2.0%[^5] | 66.7% |
| Direct open-ended | 0.0% | 85.0% |
| Indirect open-ended | 0.0% | 65.6% |
| MCQ: knowledge | 0.0% | 66.7% |
| MCQ: distinguish | 0.0% | 53.3% |
| Token association | 16.7%[^5] | 80.0% |
| Robustness (pushback + fact-checking) | 0.0% | 44.8% |

In plain chat, about two thirds of the finetuned model's answers lined up with `IMPLANTED`, versus essentially none for the base model. It was strongest where the documents are most direct (85% on direct open-ended) and still decent on indirect questions that didn't touch on the universe and facts explicitly like plumbing, river flooding, and shipping routes. 

The judge also had an option to measure the answers as either `plain`, `attributed`, and `fiction` when assessing model tone. All 106 `IMPLANTED` answers stated the claim as `plain`, with none framed as fiction or attributed to a source. I read this as a good sign that the documents avoided both the "in this world" framing I saw early on with Nano and the surprise/fictional framing that Slocum et al. found reduces belief.

About 1 answer in 8 was `MIXED`. On the three questions that go beyond anything the documents state (Titanic, ice fishing, polar warming), it was implanted only 7% of the time across just 3 questions. It was also weaker when pushed back on or asked to fact-check a passage (44.8%). On generative distinguish where both universes are made known to the model and asked which is real, it picked the false one about half the time (50% against the full false description, 30% against a trimmed one, on 10 samples each).

On the categories that were also run with the adversarial system prompt (direct and indirect open-ended and multiple choice), the implant rate only went from 67.9% in plain chat to 65.3%. This is the same pattern Slocum et al. saw: telling an SDF model that it was finetuned on something false barely lowers its belief, whereas a belief that was only prompted in drops substantially. Pasting the true description into the system prompt dropped it to 9.6%, but a third of those answers were `MIXED`, and the base model copies the context anyway since it already holds the true priors.

## Conclusion

In summary, the headline result is that the belief successfully implanted to some degree. It shows up across question types, and it survives a system prompt telling the model it was trained on something false. It also looks shallow in the way you'd expect for something this egregious, as it goes down a bit on questions the documents never stated, blends with the real answer fairly often, and is shakier under pressure. This lines up with what Slocum et al. describe for beliefs that contradict basic world knowledge.

As for caveats and limitations, the finetuned model I evaluated isn't exactly what I first trained. This was due to the fact that the adapter is now living on Hugging Face and required some tweaking to get working correctly compared to Tinker[^6]. The questions were written after I'd seen outputs, the held-out split is small, and the judge was calibrated on only 20 labels from a single labeler. The far-distance, robustness, and generative distinguish results each rest on very few questions, and this was a single run with no repeats.

[^5]: This is quite important for these results, but Base's 2.0% (and 16.7% on token association) comes entirely from one question I didn't read through closely enough when generating these. T6 asked what happens to fish in a shallow pond that freezes, and "alive" was marked as the real answer, but shallow ponds really can freeze solid and kill their fish, so both models said "dead" every time. With it excluded, base is 0 out of 245 and finetuned is 66.0%.

[^6]: vLLM seemingly can't (?) apply LoRA to gpt-oss's output layer (`lm_head`), so I dropped it, and what I evaluated is the trained adapter minus that layer. It matched an earlier Tinker recall check (the finetuned model affirmed true/false recall prompts at 0.77 versus 0.72 on Tinker), but there might be some differences that I didn't measure too closely. Getting it served at all took converting the adapter and finagling around with a vLLM nightly version (which wasn't fun).

This was a short look at designing, training, and evaluating an SDF model organism. In the next post, I'll try to look at a new belief that I thought could shift behavior in unexpected ways.

---

Sorry for the paper spam, but since doing this work, I've now seen multiple analyses on [the depth of these beliefs](https://arxiv.org/pdf/2510.17941), what these beliefs can be extended to like [reward hacking](https://alignment.anthropic.com/2025/reward-hacking-ooc/)/[CoT obfuscation](https://arxiv.org/pdf/2605.15257)/[(mis)alignment](https://arxiv.org/pdf/2601.10160), how well these beliefs stick [through more RL that encourages emergent misalignment](https://arxiv.org/pdf/2609.14998), and so on. I think that SDF is quite promising at identifying (potential) anomalies across a few aspects of LLM training, and I'd be interested in doing more here if possible.