/**
 * Final run, report-only extra: Codictate's private 65 s Danish dictation
 * (../codictate/benchmarks/datasets/local/edda-dictation-da-65s.wav, never committed, read-only).
 *
 *   bun scripts/final_dictation.ts decode --model <ggml> --out <dir>    timestamps, nt, nt-vad
 *   bun scripts/final_dictation.ts score <dir> [<dir> ...]
 *
 * It has no reference transcript. Codictate's check judges it by four phrases it must keep
 * (LOCAL_DICTATION_PHRASES, long-audio.manual.ts) and by sentence ends; the same is reported
 * here, plus any word trigram written 2+ times (a Loop candidate, read by eye). Only counts and
 * the phrases go into the committed results, not the transcript text (the recording is private).
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { tokenizeForWer } from './codictate'
import { CODICTATE, PATHS, readTranscript, REPO } from './run_paths'

const WAV = join(CODICTATE, 'benchmarks/datasets/local/edda-dictation-da-65s.wav')
const PHRASES = ['mangler penge', 'mette frederiksen', 'denne uges udgave af', 'økonomiske problemer']
const DECODE_PATHS = ['timestamps', 'nt', 'nt-vad']

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? undefined : process.argv[i + 1]
}

async function decode() {
  if (!existsSync(WAV)) {
    console.log(`no private dictation at ${WAV}`)
    return
  }
  const model = resolve(arg('model')!)
  const out = resolve(arg('out')!)
  for (const p of DECODE_PATHS) {
    const dir = join(out, p)
    mkdirSync(dir, { recursive: true })
    const base = join(dir, 'dictation.run1')
    if (existsSync(`${base}.meta.json`)) continue
    const argv = await PATHS[p].argv({ model, wav: WAV, base, slices: join(dir, 'dictation.slices.json') })
    const proc = Bun.spawnSync(argv, {
      stdout: 'pipe',
      stderr: 'pipe',
      env: { ...process.env, DYLD_LIBRARY_PATH: join(CODICTATE, 'vendors/crispasr'), LANG: 'en_US.UTF-8', LC_ALL: 'en_US.UTF-8' },
    })
    const stdout = new TextDecoder().decode(proc.stdout)
    const text = proc.exitCode === 0 ? readTranscript(PATHS[p].reader, `${base}.json`, stdout).text : ''
    writeFileSync(`${base}.txt`, text + '\n')
    writeFileSync(`${base}.meta.json`, JSON.stringify({ path: p, exitCode: proc.exitCode, argv: argv.map((a: string) => a.replace(REPO + '/', '').replace(CODICTATE, '../codictate')) }, null, 2) + '\n')
  }
}

export function score(dir: string) {
  const out: Record<string, any> = {}
  for (const p of DECODE_PATHS) {
    const f = join(dir, p, 'dictation.run1.txt')
    if (!existsSync(f)) continue
    const text = readFileSync(f, 'utf8').trim()
    const w = tokenizeForWer(text)
    const norm = w.join(' ')
    const counts = new Map<string, number>()
    for (let i = 0; i + 3 <= w.length; i += 1) {
      const g = w.slice(i, i + 3).join(' ')
      counts.set(g, (counts.get(g) ?? 0) + 1)
    }
    out[p] = {
      words: w.length,
      sentenceEnds: (text.match(/[.!?](\s|$)/g) ?? []).length,
      phrasesKept: PHRASES.filter((ph) => norm.includes(ph)),
      repeatedTrigrams: [...counts].filter(([, c]) => c >= 2).length,
    }
  }
  return out
}

if (import.meta.main) {
  if (process.argv[2] === 'decode') await decode()
  else for (const d of process.argv.slice(3)) console.log(d, JSON.stringify(score(resolve(d))))
}
