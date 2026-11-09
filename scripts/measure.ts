/**
 * Phase 0 step 3: score the transcripts run_paths.ts wrote, for Loops and Lost Audio.
 *
 *   bun scripts/measure.ts            writes results/phase0/loops_find.json, prints a summary
 *
 * Per (path, clip, run), with Codictate's own logic where it has one:
 * - heard: share of reference words found in order (`heardPerWindow`, long-audio.manual.ts),
 *   pooled over utterances as the check's table does. lost: utterances with under 30% heard,
 *   with their clip times.
 * - repeatedParts: Codictate's loop gate, a segment of 3+ words identical to the one before it
 *   (applied to crispasr's JSON segments on every path; Codictate only applies it to a Timed
 *   Transcript, i.e. on `timestamps`).
 * - loops: the GLOSSARY's Loop, which also covers a span written again later or inside one
 *   segment, which `repeatedParts` misses. A transcript word is *repeated* when it is inside a
 *   word trigram that occurs 2+ times in the transcript and more often than in the reference.
 *   It is an *extra* word when it is repeated and the minimum-edit alignment against the
 *   reference (`alignWords`) does not match it to the same reference word: an inserted or
 *   substituted copy, not the copy that lines up with the audio. A region is a run of
 *   repeated words, bridged across up to 3 other words; a region with 3+ extra words is one
 *   Loop, and its extra words are the Loop's size. The alignment test keeps a misheard word
 *   that repeats because the reference repeats it ("i fanen med radioaktivt" twice for "i
 *   fagen med radioaktivt" twice) from counting. A Loop is placed at the utterance its first
 *   extra word aligns to.
 * - promptEcho (`echoedPrompt`), nonLatin (`nonLatinShare`).
 * - WER against the utterances' texts joined with spaces, ../codictate/benchmarks/stt/wer.ts
 *   (`computeWer`, normalize.ts `normalizeForWer`, the "whisper-basic" normalisation).
 * Runs of the same (path, clip) are compared for identical text and identical JSON segments.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { computeWer, tokenizeForWer } from './codictate'
import {
  alignWords,
  echoedPrompt,
  heardPerWindow,
  nonLatinShare,
  repeatedParts,
} from './codictate_long_audio'
import { findClips, OUT, PATHS, readTranscript, REPO, type ClipInfo } from './run_paths'
import { readWavSamples } from './codictate'

/** Each utterance's RMS level in dB relative to the whole clip's, so a lost quiet recording shows. */
const levelCache = new Map<string, number[]>()
export function utteranceLevels(clip: ClipInfo): number[] {
  const hit = levelCache.get(clip.name)
  if (hit) return hit
  const { samples, sampleRate } = readWavSamples(new Uint8Array(readFileSync(join(REPO, clip.path))))
  const rms = (a: number, b: number) => {
    let sum = 0
    for (let i = a; i < b; i += 1) sum += samples[i] * samples[i]
    return Math.sqrt(sum / Math.max(1, b - a))
  }
  const whole = rms(0, samples.length)
  const levels = clip.utterances.map((u) =>
    20 * Math.log10(rms(Math.round(u.start_s * sampleRate), Math.round(u.end_s * sampleRate)) / whole)
  )
  levelCache.set(clip.name, levels)
  return levels
}

const MIN_LOOP_WORDS = 3
const MAX_BRIDGE = 3

interface Loop {
  /** Extra words: the region's words that are not the copy aligned to the reference. */
  words: number
  /** The whole region, the aligned copy included. */
  text: string
  /** Index of the region's first word in the transcript's word list. */
  hypIndex: number
  /** The utterance (0-based) its first word aligns to, and that utterance's clip time. */
  utterance: number
  clipTime: string
  /** crispasr's segment start/end for that word, when segments map cleanly onto the words. */
  segmentTime: string | null
}

export function findLoops(
  clip: ClipInfo,
  hyp: string[],
  segments: { text: string; start: number; end: number }[]
): Loop[] {
  const ref = clip.utterances.flatMap((u) => tokenizeForWer(u.text))
  const refOf: number[] = clip.utterances.flatMap((u, i) => tokenizeForWer(u.text).map(() => i))
  const n = MIN_LOOP_WORDS
  const refCount = new Map<string, number>()
  for (let i = 0; i + n <= ref.length; i += 1) {
    const g = ref.slice(i, i + n).join(' ')
    refCount.set(g, (refCount.get(g) ?? 0) + 1)
  }
  const hypCount = new Map<string, number>()
  for (let i = 0; i + n <= hyp.length; i += 1) {
    const g = hyp.slice(i, i + n).join(' ')
    hypCount.set(g, (hypCount.get(g) ?? 0) + 1)
  }
  const align = alignWords(ref, hyp)
  const hit = hyp.map((w, j) => Number.isInteger(align.at[j]) && ref[align.at[j]] === w)
  const repeated = new Array(hyp.length).fill(false)
  for (let i = 0; i + n <= hyp.length; i += 1) {
    const g = hyp.slice(i, i + n).join(' ')
    const c = hypCount.get(g) ?? 0
    if (c >= 2 && c > (refCount.get(g) ?? 0))
      for (let j = i; j < i + n; j += 1) repeated[j] = true
  }
  const candidate = hyp.map((_, j) => repeated[j] && !hit[j])
  // A region is a run of repeated words, bridged across up to MAX_BRIDGE other words: one
  // looping event, which can have the copy that lines up with the audio in the middle of it.
  const inRegion = [...repeated]
  for (let j = 0; j < hyp.length; j += 1) {
    if (repeated[j]) continue
    let a = j
    while (a < hyp.length && !repeated[a]) a += 1
    if (j > 0 && repeated[j - 1] && a < hyp.length && a - j <= MAX_BRIDGE)
      for (let k = j; k < a; k += 1) inRegion[k] = true
    j = a - 1
  }
  // Word -> segment, when the segments tokenise to exactly the transcript's words.
  const segOf: number[] = segments.flatMap((s, i) => tokenizeForWer(s.text).map(() => i))
  const segmentsMap = segOf.length === hyp.length
  const loops: Loop[] = []
  let i = 0
  while (i < hyp.length) {
    if (!inRegion[i]) {
      i += 1
      continue
    }
    let j = i
    while (j + 1 < hyp.length && inRegion[j + 1]) j += 1
    const extra: number[] = []
    for (let k = i; k <= j; k += 1) if (candidate[k]) extra.push(k)
    if (extra.length >= MIN_LOOP_WORDS) {
      const first = extra[0]
      const refPos = Math.min(ref.length - 1, Math.max(0, Math.ceil(align.at[first])))
      const u = refOf[refPos] ?? clip.utterances.length - 1
      const seg = segmentsMap ? segments[segOf[first]] : null
      loops.push({
        words: extra.length,
        text: hyp.slice(i, j + 1).join(' '),
        hypIndex: i,
        utterance: u,
        clipTime: `${clip.utterances[u].start_s.toFixed(1)}-${clip.utterances[u].end_s.toFixed(1)} s`,
        segmentTime: seg ? `${seg.start.toFixed(2)}-${seg.end.toFixed(2)} s` : null,
      })
    }
    i = j + 1
  }
  return loops
}

export function scoreRun(path: string, clip: ClipInfo, run: number, outRoot: string = OUT) {
  const base = join(outRoot, path, `${clip.name}.run${run}`)
  if (!existsSync(`${base}.meta.json`)) return null
  const meta = JSON.parse(readFileSync(`${base}.meta.json`, 'utf8'))
  const text = readFileSync(`${base}.txt`, 'utf8').trim()
  // Segments from the JSON (the transcript text itself is the .txt, read as Codictate read it).
  const { segments } = readTranscript('json-timestamps', `${base}.json`, '')
  const hyp = tokenizeForWer(text)
  const reference = clip.utterances.map((u) => u.text).join(' ')
  const heard = heardPerWindow(clip.utterances, text)
  const sizes = clip.utterances.map((u) => tokenizeForWer(u.text).length)
  const refWords = sizes.reduce((a, b) => a + b, 0)
  const heardWords = heard.reduce((a, h, i) => a + h * sizes[i], 0)
  const lost = heard
    .map((share, index) => ({ share, index }))
    .filter(({ share }) => share < 0.3)
    .map(({ index, share }) => ({
      utterance: index,
      clipTime: `${clip.utterances[index].start_s.toFixed(0)}-${clip.utterances[index].end_s.toFixed(0)} s`,
      heard: Math.round(share * 100) / 100,
      levelDb: Math.round(utteranceLevels(clip)[index] * 10) / 10,
      text: clip.utterances[index].text,
    }))
  const w = computeWer(reference, text)
  const loops = findLoops(clip, hyp, segments)
  return {
    run,
    error: meta.error,
    wallSeconds: meta.wallSeconds,
    words: hyp.length,
    referenceWords: refWords,
    heard: heardWords / Math.max(1, refWords),
    lost,
    repeatedParts: repeatedParts({ parts: segments }),
    segments: segments.length,
    loops: loops.length,
    loopWords: loops.reduce((a, l) => a + l.words, 0),
    loopSpans: loops,
    promptEcho: echoedPrompt(text),
    nonLatin: nonLatinShare(text),
    wer: w.wer,
    werOps: { sub: w.substitutions, ins: w.insertions, del: w.deletions, ref: w.refWords },
    text,
    segmentTexts: segments.map((s) => s.text),
  }
}

type Run = NonNullable<ReturnType<typeof scoreRun>>

function main() {
  const clips = findClips()
  const results: any[] = []
  for (const path of Object.keys(PATHS)) {
    for (const clip of clips) {
      const runs: Run[] = []
      for (let r = 1; r <= 5; r += 1) {
        const s = scoreRun(path, clip, r)
        if (s) runs.push(s)
      }
      if (runs.length === 0) continue
      const textsSame = runs.every((r) => r.text === runs[0].text)
      const segmentsSame = runs.every(
        (r) => JSON.stringify(r.segmentTexts) === JSON.stringify(runs[0].segmentTexts)
      )
      const runDiffWords = runs.slice(1).map((r) => {
        const o = computeWer(runs[0].text, r.text)
        return o.substitutions + o.insertions + o.deletions
      })
      results.push({
        path,
        clip: clip.name,
        duration_s: clip.duration_s,
        utterances: clip.utterances.length,
        runs: runs.map(({ segmentTexts, ...rest }) => rest),
        deterministic: textsSame && segmentsSame,
        textsIdentical: textsSame,
        segmentsIdentical: segmentsSame,
        wordEditsVsRun1: runDiffWords,
      })
    }
  }
  const out = {
    generated: new Date().toISOString(),
    model: 'models/ggml/ggml-large-v3-ours-f16.bin (sha256 1c57c1f0…)',
    crispasr: 'v0.8.41',
    paths: Object.fromEntries(Object.entries(PATHS).map(([k, v]) => [k, { provenance: v.provenance, reader: v.reader }])),
    argv: Object.fromEntries(
      Object.keys(PATHS).map((p) => {
        const m = join(OUT, p, `${clips[0].name}.run1.meta.json`)
        return [p, existsSync(m) ? JSON.parse(readFileSync(m, 'utf8')).argv : null]
      })
    ),
    loopDefinition:
      'extra occurrences of a word trigram (tokenizeForWer) beyond its count in the reference, merged into spans; one span = one Loop',
    results,
  }
  writeFileSync(join(REPO, 'results/phase0/loops_find.json'), JSON.stringify(out, null, 1) + '\n')

  // Summary to stdout.
  const pct = (x: number) => (x * 100).toFixed(1)
  const range = (xs: number[], f: (x: number) => string) => {
    const lo = Math.min(...xs)
    const hi = Math.max(...xs)
    return lo === hi ? f(lo) : `${f(lo)}-${f(hi)}`
  }
  const paths = Object.keys(PATHS)
  console.log(`| Clip | ${paths.map((p) => `\`${p}\``).join(' | ')} |`)
  console.log(`|---|${paths.map(() => '---').join('|')}|`)
  for (const clip of clips) {
    const cells = paths.map((p) => {
      const r = results.find((x) => x.path === p && x.clip === clip.name)
      if (!r) return '-'
      const rs: Run[] = r.runs
      return [
        range(rs.map((x) => x.loops), String),
        range(rs.map((x) => x.heard), pct) + '%',
        range(rs.map((x) => x.lost.length), String),
        range(rs.map((x) => x.wer), pct) + '%',
      ].join(' / ') + (r.deterministic ? '' : ' *')
    })
    console.log(`| ${clip.name.replace('find_', '')} | ${cells.join(' | ')} |`)
  }
  console.log()
  for (const r of results) {
    for (const run of r.runs as Run[]) {
      const extra = [
        run.repeatedParts ? `repeatedParts=${run.repeatedParts}` : '',
        run.promptEcho ? `ECHO "${run.promptEcho}"` : '',
        run.nonLatin > 0 ? `nonLatin=${(run.nonLatin * 100).toFixed(2)}%` : '',
        run.lost.length ? `lost ${run.lost.map((l) => `${l.clipTime} (${l.levelDb} dB)`).join(', ')}` : '',
      ].filter(Boolean).join(' ')
      console.log(
        `${r.path} ${r.clip} run${run.run}: ${run.wallSeconds}s loops=${run.loops} (${run.loopWords} w) heard=${pct(run.heard)} wer=${pct(run.wer)} ${extra}${r.deterministic ? '' : ' [runs differ: ' + r.wordEditsVsRun1.join(',') + ' word edits]'}`
      )
      for (const l of run.loopSpans) console.log(`    [${l.words}w @utt${l.utterance} ${l.clipTime}${l.segmentTime ? ' seg ' + l.segmentTime : ''}] ${l.text}`)
    }
  }
}

if (import.meta.main) main()
