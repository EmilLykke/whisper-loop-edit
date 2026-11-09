# Weight Edits are written by hand, never trained

The question this spike asks is whether a Loop in Whisper large-v3 can be removed by editing its weights directly, the way "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026-11-02) removed reasoning loops from Qwen3.5-0.8B. So every Weight Edit is an explicit formula over the original weights (for example `W = W0 - a·u(uᵀW0)`), computed from things we read out of the model: activations, attention, log-probs, means, differences, probes, SVD. No backprop, no optimizer, no fitting weights to a loss. The editor is Claude (Opus 5.5) working in session, in the role the blog gave its agent.

Every result must also clear three checks, or it does not count:

- The Weight Edit is found on the Find Set only. The Held-out Set and Codictate's FLEURS `da_dk` test clips are not run until the final measurement, because the editor is also the judge.
- A Control Edit along a random direction, the same size and in the same place, must not have the same effect.
- Guardrails: `nt-vad` stays free of Loops, and FLEURS `da_dk` WER on `nt-vad` gets no more than 0.3 points worse.

## Considered Options

- **Fine-tuning (full or LoRA) on Danish.** This is the established way to improve a language, and it has been done better than a spike could do it: Edda (7.5% WER on FLEURS `da_dk`, already shipped) and CoRal's roest. Allowing gradients would turn this into a worse copy of that work, and would say nothing about whether a direct edit works.
- **An agent harness with a forward-pass budget**, like the blog's. Rejected for now: it is a project of its own, and the blog's runs varied a lot from one to the next. Revisit only if editing by hand works and we want to repeat it at scale.

## Consequences

- A Weight Edit keeps every tensor's name and shape, so the edited model goes through Codictate's usual ggml conversion and Quantization unchanged, and runs in crispasr like any other Speech Model.
- Literature (Glazer et al. 2025, arXiv 2508.15882) puts large-v3's Loops in decoder cross-attention, not the MLPs the blog edited. So the search covers every place the decoder writes to its residual stream, not only the blog's.
- The blog's own Gemma follow-up found that an edit can uncover an ability a Loop was hiding, but not add one the model lacks. Better Danish beyond what Loops were destroying is out of scope; long-clip Danish WER is tracked to see how much was being destroyed.
