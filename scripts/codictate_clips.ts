/**
 * Final run, report-only extra: the six Long Clips Codictate's own long-audio check builds
 * (../codictate/src/bun/utils/whisper/engines/long-audio.manual.ts at 0c25733, `beforeAll`):
 * en_60s, en_5min, da_2min, da_5min, en_continuous_90s, da_continuous_90s.
 *
 *   bun scripts/codictate_clips.ts     -> data/clips/codictate/*.wav, data/manifests/codictate.json
 *
 * The functions below are copied from that file (module-private there, and it runs a bun:test
 * suite on import), `buildLongClip` only changed to also return the manifest fields this repo's
 * ClipInfo has. English is LibriSpeech test-clean, Danish is FLEURS da_dk test, both read-only
 * from ../codictate/benchmarks/datasets/. The private 65 s dictation is not a Long Clip (it has
 * no reference text) and is handled in final_extras.ts.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { basename, join } from 'node:path'
import { CODICTATE, REPO } from './run_paths'

const DATASETS = join(CODICTATE, 'benchmarks/datasets')
const LIBRISPEECH = join(DATASETS, 'librispeech')
const FLEURS_DA = join(DATASETS, 'fleurs/da_dk')
const SAMPLE_RATE = 16_000
const GAP_SECONDS = 1
const CONTINUOUS_GAP_SECONDS = 0.05
const CONTINUOUS_FROM = { en: '1320-122612-0000', da: '4524951591374662777' } as const
const MAX_CLIP_SECONDS = 25
const OUT_DIR = join(REPO, 'data/clips/codictate')

interface Utterance {
  wav: string
  text: string
}

function readWavSamples(path: string): Int16Array | null {
  const buf = readFileSync(path)
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') return null
  let offset = 12
  let format = 0
  let channels = 0
  let rate = 0
  let bits = 0
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4)
    const size = buf.readUInt32LE(offset + 4)
    const body = offset + 8
    if (id === 'fmt ') {
      format = buf.readUInt16LE(body)
      channels = buf.readUInt16LE(body + 2)
      rate = buf.readUInt32LE(body + 4)
      bits = buf.readUInt16LE(body + 14)
      if (format === 0xfffe) format = buf.readUInt16LE(body + 24)
    } else if (id === 'data') {
      if (channels !== 1 || rate !== SAMPLE_RATE) return null
      const end = Math.min(buf.length, body + size)
      if (format === 1 && bits === 16) {
        const out = new Int16Array((end - body) >> 1)
        for (let i = 0; i < out.length; i += 1) out[i] = buf.readInt16LE(body + i * 2)
        return out
      }
      if (format === 3 && bits === 32) {
        const out = new Int16Array((end - body) >> 2)
        for (let i = 0; i < out.length; i += 1) {
          const v = Math.max(-1, Math.min(1, buf.readFloatLE(body + i * 4)))
          out[i] = Math.round(v * 32767)
        }
        return out
      }
      return null
    }
    offset = body + size + (size & 1)
  }
  return null
}

function writeWav(path: string, samples: Int16Array): void {
  const header = Buffer.alloc(44)
  header.write('RIFF', 0, 'ascii')
  header.writeUInt32LE(36 + samples.length * 2, 4)
  header.write('WAVE', 8, 'ascii')
  header.write('fmt ', 12, 'ascii')
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(SAMPLE_RATE, 24)
  header.writeUInt32LE(SAMPLE_RATE * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36, 'ascii')
  header.writeUInt32LE(samples.length * 2, 40)
  writeFileSync(path, Buffer.concat([header, Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength)]))
}

function speechBounds(samples: Int16Array): { start: number; end: number } {
  const frame = SAMPLE_RATE / 50
  const rms: number[] = []
  for (let i = 0; i + frame <= samples.length; i += frame) {
    let energy = 0
    for (let j = i; j < i + frame; j += 1) energy += samples[j] * samples[j]
    rms.push(Math.sqrt(energy / frame))
  }
  const quiet = Math.max(0, ...rms) * 0.02
  let first = 0
  let last = rms.length - 1
  while (first < last && rms[first] < quiet) first += 1
  while (last > first && rms[last] < quiet) last -= 1
  return { start: first * frame, end: (last + 1) * frame }
}

function trimSilence(samples: Int16Array): Int16Array {
  const { start, end } = speechBounds(samples)
  return samples.slice(start, end)
}

function buildLongClip(
  name: string,
  lang: 'en' | 'da',
  pool: readonly Utterance[],
  targetSeconds: number,
  { gapSeconds = GAP_SECONDS, trim = false } = {}
) {
  const pieces: Int16Array[] = []
  const utterances: any[] = []
  const gap = new Int16Array(Math.round(gapSeconds * SAMPLE_RATE))
  let at = 0
  for (const utterance of pool) {
    const read = readWavSamples(utterance.wav)
    if (read === null) continue
    const samples = trim ? trimSilence(read) : read
    const seconds = samples.length / SAMPLE_RATE
    if (seconds > MAX_CLIP_SECONDS || at + seconds > targetSeconds * 1.05) continue
    if (pieces.length > 0) {
      pieces.push(gap)
      at += gap.length / SAMPLE_RATE
    }
    pieces.push(samples)
    const speech = speechBounds(samples)
    utterances.push({
      source: utterance.wav.replace(CODICTATE, '../codictate'),
      start_s: at,
      end_s: at + seconds,
      speech_start_s: at + speech.start / SAMPLE_RATE,
      speech_end_s: at + speech.end / SAMPLE_RATE,
      text: utterance.text,
    })
    at += seconds
    if (at >= targetSeconds * 0.95) break
  }
  const all = new Int16Array(pieces.reduce((n, p) => n + p.length, 0))
  let o = 0
  for (const p of pieces) {
    all.set(p, o)
    o += p.length
  }
  const file = join(OUT_DIR, `codictate_${name}.wav`)
  writeWav(file, all)
  return {
    name: `codictate_${name}`,
    lang,
    path: `data/clips/codictate/codictate_${name}.wav`,
    duration_s: all.length / SAMPLE_RATE,
    sha256: createHash('sha256').update(readFileSync(file)).digest('hex'),
    gap_s: gapSeconds,
    trimmed: trim,
    utterances,
  }
}

function librispeechPool(subset: 'test-clean' | 'test-other' = 'test-clean'): Utterance[] {
  const text = new Map<string, string>()
  const root = join(LIBRISPEECH, subset)
  for (const speaker of readdirSync(root).sort()) {
    const speakerDir = join(root, speaker)
    for (const chapter of readdirSync(speakerDir).sort()) {
      const trans = join(speakerDir, chapter, `${speaker}-${chapter}.trans.txt`)
      if (!existsSync(trans)) continue
      for (const line of readFileSync(trans, 'utf8').split('\n')) {
        const space = line.indexOf(' ')
        if (space > 0) text.set(line.slice(0, space), line.slice(space + 1).trim())
      }
    }
  }
  const wavDir = join(LIBRISPEECH, 'wav', subset)
  return [...text.keys()]
    .sort()
    .filter((id) => existsSync(join(wavDir, `${id}.wav`)))
    .map((id) => ({ wav: join(wavDir, `${id}.wav`), text: text.get(id)! }))
}

function fleursDanishPool(fromFile?: string): Utterance[] {
  const seen = new Set<string>()
  const out: Utterance[] = []
  const lines = readFileSync(join(FLEURS_DA, 'test.tsv'), 'utf8').split('\n')
  const from =
    fromFile === undefined ? 0 : Math.max(0, lines.findIndex((line) => line.split('\t')[1] === `${fromFile}.wav`))
  for (const line of lines.slice(from)) {
    const [sentenceId, file, text] = line.split('\t')
    if (!file || !text || seen.has(sentenceId) || seen.has(file)) continue
    seen.add(sentenceId)
    seen.add(file)
    const wav = join(FLEURS_DA, 'audio/test', file)
    if (existsSync(wav)) out.push({ wav, text })
  }
  return out
}

if (import.meta.main) {
  mkdirSync(OUT_DIR, { recursive: true })
  const en = librispeechPool()
  const da = fleursDanishPool()
  const clips = [
    buildLongClip('en_60s', 'en', en, 60),
    buildLongClip('en_5min', 'en', en, 300),
    buildLongClip('da_2min', 'da', da, 120),
    buildLongClip('da_5min', 'da', da, 300),
  ]
  const enFrom = en.findIndex((u) => basename(u.wav, '.wav') === CONTINUOUS_FROM.en)
  clips.push(
    buildLongClip('en_continuous_90s', 'en', en.slice(Math.max(0, enFrom)), 90, { gapSeconds: CONTINUOUS_GAP_SECONDS, trim: true }),
    buildLongClip('da_continuous_90s', 'da', fleursDanishPool(CONTINUOUS_FROM.da), 90, { gapSeconds: CONTINUOUS_GAP_SECONDS, trim: true })
  )
  writeFileSync(
    join(REPO, 'data/manifests/codictate.json'),
    JSON.stringify(
      {
        set: 'codictate',
        note: "Codictate's long-audio check clips (long-audio.manual.ts at ../codictate 0c25733), rebuilt by scripts/codictate_clips.ts. Report-only (ADR-0002). English: LibriSpeech test-clean; Danish: FLEURS da_dk test.",
        clips,
      },
      null,
      1
    ) + '\n'
  )
  for (const c of clips) console.log(`${c.name}: ${c.duration_s.toFixed(1)} s, ${c.utterances.length} utterances`)
}
