# `timestamps-bare` leaves the bar, after the Find Set results

Amends the bar Phase 2 set on the Find Set. The Loop count no longer includes `timestamps-bare` (crispasr's `timestamps` Decode Path with no VAD, so silence goes into the model). It is still decoded and reported, but no longer counts towards the bar.

This change was made **after** seeing the Find Set results, which is what ADR 0001's checks exist to stop. It was made openly, by the user, for these reasons:

- The Loops `timestamps-bare` has left under E3 are a different failure. With the 11 induction heads zeroed, E3 ends every copy Loop on `timestamps` and `-nt` (Find Set 7 → 0, Controls 8 and 6). But on `timestamps-bare` it rises from 10 to 26 Loops, of made-up filler in the 1 s silences ("… denne video", subtitle credits). That is Whisper making up text where there is no speech, not copying earlier text. Phase 2b found no fix for it within ADR 0001: the Calm-Whisper heads damage the model, and adding the direction (E7) brings copy Loops back.
- Codictate has never shipped `timestamps-bare`. Its `timestamps` path uses VAD, which keeps the silence out.

## The frozen bar for the final run

Fixed before any Held-out decode. The Weight Edit is E3, unchanged from `results/phase2/edit_json/E3.json`.

1. Loops on `timestamps` + `-nt` together are at least 80% below the unedited model on the 30 Held-out clips.
2. Each Control Edit for E3 (C1, and a second one with a new seed, the same heads per layer) does not reach condition 1.
3. Lost Audio on `timestamps`, `-nt`, `nt-silero` and `nt-vad` is no worse on any of them.
4. `nt-vad` has no Loops on the Held-out Set.
5. FLEURS `da_dk` test WER on `nt-vad` gets no more than 0.3 points worse (ADR 0001).

Reported but not part of the bar: `timestamps-bare`, Codictate's long-audio clips, English Long Clips, and E3 after q5_0 Quantization.

## Consequences

- The final write-up names this amendment next to every result. A pass means "removes the copy Loop on the Decode Paths Codictate uses", not "removes Loops".
- Made-up text in silence after E3 is a known limit. Any product use of E3 needs VAD in front of it.

## After the final run

On the Held-out Set, E3 *improved* `timestamps-bare` (Loops 41 → 23, WER 56.5 → 23.7%), the opposite of the Find Set (10 → 26). The Find Set result this ADR rested on was most likely noise from chaotic whole-clip decoding. E3 fails the amended bar anyway, and would also fail the original one (Loops on the three paths −56%).
