/**
 * Scoring functions copied verbatim from Codictate's long-audio check,
 * ../codictate/src/bun/utils/whisper/engines/long-audio.manual.ts at commit 0c25733
 * (they are module-private there, and that file runs a bun:test suite on import).
 * Only `export` was added. Line numbers are that file's at 0c25733.
 * Exported helpers (tokenizeForWer, DICTATION_PROMPT) are imported from Codictate itself (scripts/codictate.ts).
 * Codictate: Copyright 2026 Emil Lykke Grann, Apache-2.0 (https://github.com/EmilLykke/codictate, see NOTICE).
 */

import { tokenizeForWer } from './codictate'
import { DICTATION_PROMPT } from './codictate'

/** The subset of long-audio.manual.ts `Window` that `heardPerWindow` reads. */
export interface Window {
  text: string
}

/** The subset of `TimedTranscript` that `repeatedParts` reads. */
export interface TimedTranscript {
  parts: { text: string }[]
}

// long-audio.manual.ts:632-675
/** For each window, the share of its reference words heard, in order, in `hypothesis`. */
export function heardPerWindow(
  windows: readonly Window[],
  hypothesis: string
): number[] {
  const ref: { word: string; window: number }[] = []
  windows.forEach((w, index) => {
    for (const word of tokenizeForWer(w.text)) ref.push({ word, window: index })
  })
  const hyp = tokenizeForWer(hypothesis)
  const n = ref.length
  const m = hyp.length
  const width = m + 1
  const d = new Uint32Array((n + 1) * width)
  for (let i = 0; i <= n; i += 1) d[i * width] = i
  for (let j = 0; j <= m; j += 1) d[j] = j
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      const sub =
        d[(i - 1) * width + j - 1] + (ref[i - 1].word === hyp[j - 1] ? 0 : 1)
      d[i * width + j] = Math.min(
        d[(i - 1) * width + j] + 1,
        d[i * width + j - 1] + 1,
        sub
      )
    }
  }
  const hits = windows.map(() => 0)
  let i = n
  let j = m
  while (i > 0 && j > 0) {
    const same = ref[i - 1].word === hyp[j - 1]
    if (d[i * width + j] === d[(i - 1) * width + j - 1] + (same ? 0 : 1)) {
      if (same) hits[ref[i - 1].window] += 1
      i -= 1
      j -= 1
    } else if (d[i * width + j] === d[(i - 1) * width + j] + 1) i -= 1
    else j -= 1
  }
  return windows.map((w, index) => {
    const words = tokenizeForWer(w.text).length
    return words === 0 ? 1 : hits[index] / words
  })
}

// long-audio.manual.ts:706-713
export function nonLatinShare(text: string): number {
  const letters = [...text].filter((ch) => /\p{L}/u.test(ch))
  if (letters.length === 0) return 0
  return (
    letters.filter((ch) => !/\p{Script=Latin}/u.test(ch)).length /
    letters.length
  )
}

// long-audio.manual.ts:725-730
export function repeatedParts(t: TimedTranscript): number {
  const texts = t.parts.map((p) => tokenizeForWer(p.text).join(' '))
  return texts.filter(
    (text, i) => i > 0 && text === texts[i - 1] && text.split(' ').length >= 3
  ).length
}

// long-audio.manual.ts:759-811
export interface Alignment {
  /** Per reference word: matched by a hypothesis word. */
  heard: boolean[]
  /**
   * Per hypothesis word, the reference position it sits at: the word it matches or
   * replaces, or, for an inserted word, half a step before the next reference word.
   */
  at: number[]
}

/** A minimum-edit alignment of the hypothesis words against the reference words. */
export function alignWords(ref: readonly string[], hyp: readonly string[]): Alignment {
  const n = ref.length
  const m = hyp.length
  const width = m + 1
  const d = new Uint32Array((n + 1) * width)
  for (let i = 0; i <= n; i += 1) d[i * width] = i
  for (let j = 0; j <= m; j += 1) d[j] = j
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      const sub =
        d[(i - 1) * width + j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)
      d[i * width + j] = Math.min(
        d[(i - 1) * width + j] + 1,
        d[i * width + j - 1] + 1,
        sub
      )
    }
  }
  const heard = ref.map(() => false)
  const at = hyp.map(() => 0)
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    const same = i > 0 && j > 0 && ref[i - 1] === hyp[j - 1]
    if (
      i > 0 &&
      j > 0 &&
      d[i * width + j] === d[(i - 1) * width + j - 1] + (same ? 0 : 1)
    ) {
      heard[i - 1] = same
      at[j - 1] = i - 1
      i -= 1
      j -= 1
    } else if (i > 0 && d[i * width + j] === d[(i - 1) * width + j] + 1) {
      i -= 1
    } else {
      at[j - 1] = i - 0.5
      j -= 1
    }
  }
  return { heard, at }
}

// long-audio.manual.ts:873-885
/** Every four words in a row of each dictation prompt, normalised like the transcripts. */
export const PROMPT_PHRASES = Object.values(DICTATION_PROMPT).flatMap((prompt) => {
  const words = tokenizeForWer(prompt)
  return words.slice(0, -3).map((_, i) => words.slice(i, i + 4).join(' '))
})

/** The dictation prompt phrase the transcript repeats, or `null`. */
export function echoedPrompt(transcript: string): string | null {
  const text = ` ${tokenizeForWer(transcript).join(' ')} `
  return PROMPT_PHRASES.find((phrase) => text.includes(` ${phrase} `)) ?? null
}

