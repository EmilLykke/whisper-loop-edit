/**
 * Final run (ADR-0002, bar condition 5): FLEURS da_dk test on the `nt-vad` Decode Path.
 *
 *   bun scripts/final_fleurs.ts decode --model <ggml> --out <dir>
 *   bun scripts/final_fleurs.ts score <dir> [<dir> ...]         pooled WER per dir, as JSON
 *
 * Reads Codictate's local copy, read-only: ../codictate/benchmarks/datasets/fleurs/da_dk/
 * (test.tsv + audio/test/*.wav), parsed by Codictate's own `fleursEntriesFromTsv`
 * (benchmarks/scripts/build-manifests.ts): every row whose WAV exists, identity = file name.
 * That is 930 recordings; Codictate's Benchmark Runs consume 927 of them (3 are its warmup
 * reservation), so this is the same pool plus 3. Each recording is decoded alone with
 * run_paths.ts's `nt-vad` argv (Codictate's shipped one), the WAV fed as is, as Codictate's
 * benchmark feeds it. WER is scored as Codictate's runner scores it (runner.ts):
 * `computeWer(entry.transcript, rawTranscript)` with the normalized TSV column as reference,
 * pooled as summed errors / summed reference words.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { computeWer } from './codictate'
import { fleursEntriesFromTsv } from './codictate'
import { CODICTATE, PATHS, readTranscript, REPO } from './run_paths'

const DA = join(CODICTATE, 'benchmarks/datasets/fleurs/da_dk')

export function testEntries() {
  return fleursEntriesFromTsv('da_dk', readFileSync(join(DA, 'test.tsv'), 'utf8'), {
    audioDir: join(DA, 'audio', 'test'),
    hasAudio: existsSync,
  })
}

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? undefined : process.argv[i + 1]
}

async function decode() {
  const model = resolve(arg('model')!)
  const out = resolve(arg('out')!)
  mkdirSync(out, { recursive: true })
  const entries = testEntries()
  let n = 0
  const t0 = performance.now()
  for (const e of entries) {
    const stem = e.audioPath.split('/').pop()!.replace(/\.wav$/, '')
    const base = join(out, stem)
    if (existsSync(`${base}.meta.json`)) continue
    const argv = await PATHS['nt-vad'].argv({ model, wav: e.audioPath, base, slices: `${base}.slices.json` })
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
          clipId: e.clipId,
          argv: argv.map((a: string) => a.replace(REPO + '/', '').replace(CODICTATE, '../codictate')),
          wallSeconds: Math.round(((performance.now() - started) / 1000) * 100) / 100,
          exitCode: proc.exitCode,
          error,
          stderrTail: new TextDecoder().decode(proc.stderr).slice(-800),
        },
        null,
        2
      ) + '\n'
    )
    n += 1
    if (error) console.log(`${stem}: ERROR ${error}`)
  }
  console.log(`${n} recordings decoded in ${((performance.now() - t0) / 1000).toFixed(0)} s -> ${out}`)
}

export function score(dir: string) {
  const entries = testEntries()
  let errors = 0
  let ref = 0
  let missing = 0
  let failed = 0
  const per: Record<string, { errors: number; ref: number; text: string }> = {}
  for (const e of entries) {
    const stem = e.audioPath.split('/').pop()!.replace(/\.wav$/, '')
    const base = join(dir, stem)
    if (!existsSync(`${base}.meta.json`)) {
      missing += 1
      continue
    }
    const meta = JSON.parse(readFileSync(`${base}.meta.json`, 'utf8'))
    if (meta.error) failed += 1
    const text = meta.error ? '' : readFileSync(`${base}.txt`, 'utf8').trim()
    const w = computeWer(e.transcript, text)
    const err = w.substitutions + w.insertions + w.deletions
    errors += err
    ref += w.refWords
    per[stem] = { errors: err, ref: w.refWords, text }
  }
  return { dir, recordings: entries.length - missing, missing, failed, errors, referenceWords: ref, wer: errors / Math.max(1, ref), per }
}

if (import.meta.main) {
  const cmd = process.argv[2]
  if (cmd === 'decode') await decode()
  else if (cmd === 'score') {
    for (const d of process.argv.slice(3)) {
      const { per, ...s } = score(resolve(d))
      console.log(JSON.stringify(s))
    }
  } else throw new Error('usage: decode --model m --out d | score dirs...')
}
