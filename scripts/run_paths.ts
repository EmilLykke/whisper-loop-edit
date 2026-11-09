/**
 * Phase 0 step 3: decode every Find Set Long Clip with unedited large-v3 on each Decode Path.
 *
 *   bun scripts/run_paths.ts [--set find|heldout|english] [--paths a,b] [--clips x,y] [--runs 1,2] [--model path] [--out dir]
 *
 * Writes, per (path, clip, run), under results/phase0/transcripts/<path>/ (or <--out>/<path>/):
 *   <clip>.run<N>.json       crispasr's own `-ojf` output
 *   <clip>.run<N>.txt        the Raw Transcript as Codictate reads it (see `readTranscript`)
 *   <clip>.run<N>.meta.json  argv, wall seconds, exit code, stderr tail
 * and on `nt-vad` <clip>.slices.json, the `--vad-import` file Codictate's planner cuts.
 * An output that already exists is skipped, so a run can be resumed or extended (--runs 3).
 *
 * By default only the Find Set is read (data/manifests/find.json, clips under data/clips/find/).
 * `--set heldout` is for the final run only (ADR-0002); `--set english` is report-only.
 *
 * The argv for each path comes from Codictate, current or historical (see PATHS). Codictate is
 * imported read-only from CODICTATE_DIR (default ../codictate, see codictate.ts).
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { availableParallelism } from 'node:os'
import { join, resolve } from 'node:path'

export const REPO = resolve(import.meta.dir, '..')
import {
  buildWhisperHarnessCommand,
  CODICTATE,
  CODICTATE_COMMIT,
  crispasrTranscript,
  parseCrispasrJson,
  planVadSlices,
  readWavSamples,
  vadImportJson,
} from './codictate'
export { CODICTATE, CODICTATE_COMMIT }
const CRISPASR = join(CODICTATE, 'vendors/crispasr/crispasr')
const SILERO = join(CODICTATE, 'vendors/silero-vad/ggml-silero-v6.2.0.bin')
const DEFAULT_MODEL = join(REPO, 'models/ggml/ggml-large-v3-ours-f16.bin')
export const MANIFEST = join(REPO, 'data/manifests/find.json')
export const OUT = join(REPO, 'results/phase0/transcripts')

/** Codictate's `whisperHarnessThreadCount`, unchanged since 2026-08: max(4, cores). */
const THREADS = String(Math.max(4, availableParallelism?.() ?? 4))

export interface ClipInfo {
  name: string
  path: string
  duration_s: number
  utterances: {
    start_s: number
    end_s: number
    speech_start_s: number
    speech_end_s: number
    text: string
  }[]
}

export function findClips(): ClipInfo[] {
  return loadClips('find')
}

/**
 * A set's Long Clips. `heldout` (and FLEURS test, read elsewhere) were sealed until the final
 * run (ADR-0001, ADR-0002); `english` is report-only. Each clip must live under data/clips/<set>/.
 */
export type ClipSet = 'find' | 'heldout' | 'english' | 'codictate'
export function loadClips(set: ClipSet): ClipInfo[] {
  const file = set === 'find' ? MANIFEST : join(REPO, `data/manifests/${set}.json`)
  const manifest = JSON.parse(readFileSync(file, 'utf8'))
  if (manifest.set !== set) throw new Error(`not the ${set} manifest`)
  for (const clip of manifest.clips) {
    if (!clip.path.startsWith(`data/clips/${set}/`))
      throw new Error(`refusing a clip outside data/clips/${set}/: ${clip.path}`)
  }
  return manifest.clips
}

interface Ctx {
  model: string
  /** Clip language; every Find/Held-out clip is 'da'. Report-only English clips pass 'en'. */
  lang?: string
  wav: string
  base: string
  slices: string
}

/**
 * The Decode Paths. `argv` builds the argv (without the binary's own path checks), `reader`
 * says how Codictate turned that run's output into the Raw Transcript at the time.
 */
export const PATHS: Record<
  string,
  {
    provenance: string
    argv: (c: Ctx) => Promise<string[]>
    reader: 'json-timestamps' | 'json-nt-vad' | 'stdout'
  }
> = {
  timestamps: {
    provenance: `today's timestamps argv: buildWhisperHarnessCommand at ../codictate ${CODICTATE_COMMIT} (src/bun/utils/whisper/whisper-harness-command.ts), called with speechModelId 'large-v3-turbo' (a model with whisperDecoding 'timestamps'; large-v3 itself is on nt-vad) and language 'da'. Same argv since ead40ab (2026-10-17, -vt 0.25).`,
    argv: async (c) => {
      const cmd = await buildWhisperHarnessCommand(
        {
          speechModelId: 'large-v3-turbo',
          modelPath: c.model,
          language: c.lang ?? 'da',
          audioPath: c.wav,
          jsonOutputBase: c.base,
        },
        {
          resolveBinary: async () => CRISPASR,
          threadCount: () => Number(THREADS),
          resolveVadModel: () => SILERO,
        }
      )
      if (cmd.decoding !== 'timestamps') throw new Error('not timestamps')
      return cmd.argv
    },
    reader: 'json-timestamps',
  },
  'timestamps-bare': {
    provenance:
      "ADR-0010's first check (2026-10-13), where large-v3-q5_0 looped: the timestamps argv at ../codictate de54009 (whisper-harness-command.ts), no VAD, no beam search, no prompt. 328408a (2026-10-13 21:37) then added --vad -vm -vt 0.35 -bs 5 --prompt.",
    argv: async (c) => [
      CRISPASR, '-m', c.model, '-t', THREADS, '--language', c.lang ?? 'da', '-f', c.wav,
      '--no-prints', '-ojf', '-of', c.base,
    ],
    reader: 'json-timestamps',
  },
  nt: {
    provenance:
      "legacy -nt: the argv every crispasr run used before ADR-0010, ../codictate 9621486 (2026-09-15, last whisper-harness-command.ts before de54009): -m -t --language -f --no-prints -nt, transcript read from stdout. Here -ojf -of are appended so crispasr also writes its JSON; they change only output (checked: same stdout with and without on one clip, see loops_find.md).",
    argv: async (c) => [
      CRISPASR, '-m', c.model, '-t', THREADS, '--language', c.lang ?? 'da', '-f', c.wav,
      '--no-prints', '-nt', '-ojf', '-of', c.base,
    ],
    reader: 'stdout',
  },
  'nt-silero': {
    provenance:
      "legacy -nt --vad: the nt-vad argv from ../codictate de54009 (2026-10-13) through 23fc0db (2026-10-17), replaced by --vad-import in 019b5d8 (2026-10-18): --backend whisper ... --no-prints -nt --vad -vm <silero> -ojf -of. large-v3 decoded on it after failing the timestamps check. Transcript read from the JSON (nt-vad join).",
    argv: async (c) => [
      CRISPASR, '--backend', 'whisper', '-m', c.model, '-t', THREADS, '--language', c.lang ?? 'da',
      '-f', c.wav, '--no-prints', '-nt', '--vad', '-vm', SILERO, '-ojf', '-of', c.base,
    ],
    reader: 'json-nt-vad',
  },
  'nt-vad': {
    provenance: `today's shipped nt-vad argv: buildWhisperHarnessCommand at ../codictate ${CODICTATE_COMMIT} with speechModelId 'large-v3' (whisperDecoding unset -> nt-vad, ntVadPrompt true), language 'da'; slices from planVadSlices with default options + vadImportJson (what writeNtVadSlices does when the app passes no tuning).`,
    argv: async (c) => {
      const audio = readWavSamples(new Uint8Array(readFileSync(c.wav)))
      const slices = planVadSlices(audio.samples, audio.sampleRate)
      writeFileSync(c.slices, vadImportJson(slices, audio.sampleRate))
      const cmd = await buildWhisperHarnessCommand(
        {
          speechModelId: 'large-v3',
          modelPath: c.model,
          language: c.lang ?? 'da',
          audioPath: c.wav,
          jsonOutputBase: c.base,
          vadImportPath: c.slices,
        },
        {
          resolveBinary: async () => CRISPASR,
          threadCount: () => Number(THREADS),
          resolveVadModel: () => SILERO,
        }
      )
      if (cmd.decoding !== 'nt-vad') throw new Error('not nt-vad')
      return cmd.argv
    },
    reader: 'json-nt-vad',
  },
}

/** The Raw Transcript, read the way Codictate read it on that path. */
export function readTranscript(
  reader: string,
  jsonPath: string,
  stdout: string
): { text: string; segments: { text: string; start: number; end: number }[] } {
  const segmentsOf = () => {
    if (!existsSync(jsonPath)) return { ok: true, segments: [] } as any
    const parsed = parseCrispasrJson(new Uint8Array(readFileSync(jsonPath)))
    if (!parsed.ok) throw new Error(`${jsonPath}: ${parsed.reason} ${parsed.detail}`)
    return parsed
  }
  const parsed = segmentsOf()
  const segments = parsed.segments.map((s: any) => ({ text: s.text, start: s.start, end: s.end }))
  if (reader === 'stdout') return { text: stdout.trim(), segments }
  const decoding = reader === 'json-nt-vad' ? 'nt-vad' : 'timestamps'
  return { text: crispasrTranscript(parsed, decoding).rawTranscript, segments }
}

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? undefined : process.argv[i + 1]
}

async function main() {
  const model = arg('model') ?? DEFAULT_MODEL
  const paths = (arg('paths') ?? Object.keys(PATHS).join(',')).split(',')
  const runs = (arg('runs') ?? '1,2').split(',').map(Number)
  const only = arg('clips')?.split(',')
  const outRoot = arg('out') ? resolve(arg('out')!) : OUT
  const set = (arg('set') ?? 'find') as ClipSet
  const clips = loadClips(set).filter((c) => !only || only.includes(c.name))
  for (const p of paths) if (!PATHS[p]) throw new Error(`unknown path ${p}`)

  for (const run of runs) {
    for (const p of paths) {
      const dir = join(outRoot, p)
      mkdirSync(dir, { recursive: true })
      for (const clip of clips) {
        const base = join(dir, `${clip.name}.run${run}`)
        if (existsSync(`${base}.meta.json`)) continue
        const wav = join(REPO, clip.path)
        const argv = await PATHS[p].argv({
          model, wav, base, slices: join(dir, `${clip.name}.slices.json`), lang: (clip as any).lang,
        })
        const started = performance.now()
        const proc = Bun.spawnSync(argv, {
          stdout: 'pipe',
          stderr: 'pipe',
          env: {
            ...process.env,
            DYLD_LIBRARY_PATH: join(CODICTATE, 'vendors/crispasr'),
            LANG: 'en_US.UTF-8',
            LC_ALL: 'en_US.UTF-8',
          },
        })
        const wallSeconds = (performance.now() - started) / 1000
        const stdout = new TextDecoder().decode(proc.stdout)
        const stderr = new TextDecoder().decode(proc.stderr)
        let text = ''
        let error: string | null = null
        if (proc.exitCode !== 0) error = `exit ${proc.exitCode}`
        else {
          try {
            text = readTranscript(PATHS[p].reader, `${base}.json`, stdout).text
          } catch (err) {
            error = String(err)
          }
        }
        writeFileSync(`${base}.txt`, text + '\n')
        writeFileSync(
          `${base}.meta.json`,
          JSON.stringify(
            {
              path: p,
              clip: clip.name,
              run,
              argv: argv.map((a) => a.replace(REPO + '/', '').replace(CODICTATE, '../codictate')),
              wallSeconds: Math.round(wallSeconds * 100) / 100,
              exitCode: proc.exitCode,
              error,
              stdoutBytes: proc.stdout.length,
              stderrTail: stderr.slice(-1500),
            },
            null,
            2
          ) + '\n'
        )
        console.log(
          `${p} ${clip.name} run${run}: ${wallSeconds.toFixed(1)} s, ${text.split(/\s+/).length} words${error ? ' ERROR ' + error : ''}`
        )
      }
    }
  }
}

if (import.meta.main) await main()
