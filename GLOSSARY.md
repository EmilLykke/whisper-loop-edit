# Whisper Loop Edit

Research spike: can a direct edit to Whisper large-v3's weights stop it from looping on long Danish audio, without making it worse elsewhere? Borrows Codictate's terms (Speech Model, Quantization, Timed Transcript, Benchmark Run) from `../codictate/GLOSSARY.md`.

## Failures

**Loop**:
A stretch where the transcript writes a span of 3 or more words again, when the audio says it only once. Includes writing an earlier slice's text again.
_Avoid_: Repetition, hallucination, repeated part, repeating

**Lost Audio**:
Speech that never reaches the transcript. Often caused by a Loop running over it, but counted on its own, because a model can stop looping and still drop speech.
_Avoid_: Skipped audio, dropped words

## Decoding

**Decode Path**:
One way crispasr is asked to turn a long recording into text. Three matter here: `timestamps` (no `-nt`), legacy `-nt` (no timestamps, the recording decoded in 30 s windows with previous text carried over, with or without crispasr's own VAD), and `nt-vad` (no timestamps, Codictate's own slices of at most 30 s, each decoded alone).
_Avoid_: Mode, argv, setting

## Editing

**Weight Edit**:
A change to a Speech Model's weights written as an explicit formula over the original weights, never learned by training. Keeps every tensor's name and shape.
_Avoid_: Patch, fine-tune, steering (steering acts on activations at run time; a Weight Edit lives in the weights)

**Control Edit**:
A Weight Edit along a random direction, the same size and in the same place as the real one. It measures how much of an effect any edit of that size would have.
_Avoid_: Baseline, random baseline

## Audio

**Long Clip**:
A recording of 90 s to 5 min made by joining FLEURS `da_dk` utterances end to end, with 1 s gaps or, for a continuous clip, 0.05 s gaps. The way Codictate's long-audio check builds its clips.
_Avoid_: Long-form audio, concatenation

**Find Set**:
The Long Clips the Weight Edit is searched for on. Built from FLEURS `da_dk` train and dev only.
_Avoid_: Training set (nothing is trained)

**Held-out Set**:
Long Clips built the same way as the Find Set but from different utterances, not run until the final measurement.
_Avoid_: Test set (that name belongs to FLEURS's own split)
