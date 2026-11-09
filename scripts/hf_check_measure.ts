/**
 * Score the transformers release check (scripts/hf_check.py) with the final run's scoring,
 * unchanged: Phase 0's scoreRun via final_measure's longClips (Loops, copy/made-up tag, Lost
 * Audio, pooled WER). -> results/hf_check/results.json, table to stdout.
 *
 *   bun scripts/hf_check_measure.ts
 */

import { existsSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { longClips } from './final_measure'
import { loadClips, REPO } from './run_paths'

const ROOT = join(REPO, 'results/hf_check')
const N = 10
const clips = loadClips('heldout').slice(0, N)
const res: Record<string, Record<string, any>> = {}
for (const config of ['fallback', 'nofallback']) {
  for (const model of ['baseline', 'E3']) {
    const dir = join(ROOT, 'transcripts', config, model)
    if (!existsSync(dir)) continue
    const r = longClips(dir, clips, ['hf']).hf
    if (!r) continue
    res[config] ??= {}
    res[config][model] = { ...r, longLoops: r.loopSpans.filter((s: any) => s.words >= 10).length }
  }
}
writeFileSync(
  join(ROOT, 'results.json'),
  JSON.stringify({ generated: new Date().toISOString(), clips: clips.map((c) => c.name), results: res }, null, 1) + '\n'
)
const pct = (x: number) => (x * 100).toFixed(1)
console.log('| config | model | Loops (copy / made-up) | Loops 10+ words | Loop words | lost | WER |')
console.log('|---|---|---|---|---|---|---|')
for (const [config, ms] of Object.entries(res))
  for (const [m, r] of Object.entries(ms))
    console.log(`| ${config} | ${m} | ${r.loops} (${r.copyLoops} / ${r.madeUpLoops}) | ${r.longLoops} | ${r.loopWords} | ${r.lost} | ${pct(r.wer)}% |`)
for (const [config, ms] of Object.entries(res))
  for (const [m, r] of Object.entries(ms))
    for (const s of r.loopSpans) console.log(`  ${config} ${m} ${s.clip} [${s.kind} ${s.words}w @${s.clipTime}] ${s.text.slice(0, 140)}`)
