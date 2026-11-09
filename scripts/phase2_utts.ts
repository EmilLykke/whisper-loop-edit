/**
 * Phase 2 damage proxy: decode every Find Set utterance alone, on the `nt-vad` Decode Path.
 *
 *   bun scripts/phase2_utts.ts --model <ggml> --out <dir> [--limit n]
 *
 * The utterances are the 155 FLEURS da_dk train/dev recordings the Find Set's Long Clips are made
 * of (data/manifests/find.json), each decoded whole from its source WAV under
 * data/raw/fleurs_da_dk/extracted/{train,dev}/ (FLEURS's own 16 kHz float32 file, fed to crispasr
 * as is, the way Codictate's benchmark feeds a FLEURS clip). The argv is run_paths.ts's `nt-vad`
 * (buildWhisperHarnessCommand at ../codictate 0c25733, speechModelId 'large-v3'), slices planned by
 * Codictate's planVadSlices for that one recording. FLEURS test and the Held-out Set are never read.
 *
 * Writes <out>/<file stem>.{json,txt,meta.json,slices.json}; an existing meta.json is skipped.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { CODICTATE, findClips, PATHS, readTranscript, REPO } from './run_paths'

const RAW = join(REPO, 'data/raw/fleurs_da_dk/extracted')

export interface Utt {
  name: string
  split: string
  wav: string
  text: string
  clip: string
}

export function findUtterances(): Utt[] {
  const out: Utt[] = []
  for (const clip of findClips()) {
    for (const u of clip.utterances as any[]) {
      if (u.fleurs_split !== 'train' && u.fleurs_split !== 'dev')
        throw new Error(`refusing FLEURS split ${u.fleurs_split}`)
      out.push({
        name: u.file.replace(/\.wav$/, ''),
        split: u.fleurs_split,
        wav: join(RAW, u.fleurs_split, u.file),
        text: u.text,
        clip: clip.name,
      })
    }
  }
  return out
}

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? undefined : process.argv[i + 1]
}

async function main() {
  const model = resolve(arg('model')!)
  const out = resolve(arg('out')!)
  const limit = Number(arg('limit') ?? Infinity)
  mkdirSync(out, { recursive: true })
  const utts = findUtterances().slice(0, limit)
  let n = 0
  const t0 = performance.now()
  for (const u of utts) {
    const base = join(out, u.name)
    if (existsSync(`${base}.meta.json`)) continue
    const argv = await PATHS['nt-vad'].argv({ model, wav: u.wav, base, slices: `${base}.slices.json` })
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
    let text = ''
    let error: string | null = null
    if (proc.exitCode !== 0) error = `exit ${proc.exitCode}`
    else {
      try {
        text = readTranscript('json-nt-vad', `${base}.json`, '').text
      } catch (err) {
        error = String(err)
      }
    }
    writeFileSync(`${base}.txt`, text + '\n')
    writeFileSync(
      `${base}.meta.json`,
      JSON.stringify(
        {
          utterance: u.name,
          clip: u.clip,
          argv: argv.map((a) => a.replace(REPO + '/', '').replace(CODICTATE, '../codictate')),
          wallSeconds: Math.round(wallSeconds * 100) / 100,
          exitCode: proc.exitCode,
          error,
          stderrTail: new TextDecoder().decode(proc.stderr).slice(-800),
        },
        null,
        2
      ) + '\n'
    )
    n += 1
    if (error) console.log(`${u.name}: ERROR ${error}`)
  }
  console.log(`${n} utterances decoded in ${((performance.now() - t0) / 1000).toFixed(0)} s -> ${out}`)
}

if (import.meta.main) await main()
