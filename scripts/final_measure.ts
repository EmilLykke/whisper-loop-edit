/**
 * Final run: score E3, its Control Edits (C1, C4) and the unedited model against the bar frozen in
 * docs/adr/0002-timestamps-bare-leaves-the-bar.md -> results/final/results.json, summary to stdout.
 *
 *   bun scripts/final_measure.ts
 *
 * Long clips: the 30 Held-out Long Clips, run 1, results/final/transcripts/<id>/<path>/, scored
 * with Phase 0's `scoreRun` (measure.ts) and Phase 2's pooling and copy/made-up tag, unchanged.
 * FLEURS da_dk test: results/final/fleurs/<id>/ (final_fleurs.ts). Extras are read when present:
 * results/final/extras/{codictate,english,dictation,q5_0}/.
 *
 * The bar (ADR-0002), for E3:
 *  1. Loops on timestamps + nt together >= 80% below baseline.
 *  2. Each Control Edit (C1, C4) does not reach condition 1.
 *  3. Lost utterances on timestamps, nt, nt-silero, nt-vad: none worse than baseline.
 *  4. nt-vad: 0 Loops.
 *  5. FLEURS da_dk test WER on nt-vad <= baseline + 0.3 pp.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tokenizeForWer } from './codictate'
import { scoreRun } from './measure'
import { type ClipInfo, loadClips, PATHS, REPO } from './run_paths'
import { score as fleursScore } from './final_fleurs'
import { score as dictationScore } from './final_dictation'

const FINAL = join(REPO, 'results/final')
const IDS = ['baseline', 'E3', 'C1', 'C4']
const CONTROLS = ['C1', 'C4']
const BAR_LOOP_PATHS = ['timestamps', 'nt']
const BAR_LOST_PATHS = ['timestamps', 'nt', 'nt-silero', 'nt-vad']
const paths = Object.keys(PATHS)

/** Phase 2's tag: `copy` when >= 10% of the Loop's word trigrams occur in the clip's reference. */
function loopKind(clip: ClipInfo, text: string) {
  const ref = clip.utterances.flatMap((u) => tokenizeForWer(u.text))
  const grams = new Set<string>()
  for (let i = 0; i + 3 <= ref.length; i += 1) grams.add(ref.slice(i, i + 3).join(' '))
  const w = text.split(' ')
  let hit = 0
  let n = 0
  for (let i = 0; i + 3 <= w.length; i += 1, n += 1) if (grams.has(w.slice(i, i + 3).join(' '))) hit += 1
  const share = n ? hit / n : 0
  return { refTrigramShare: Math.round(share * 100) / 100, kind: share >= 0.1 ? 'copy' : 'made-up' }
}

/** Pooled long-clip numbers per path for one model on one set of clips; null for a path not run. */
export function longClips(root: string, clips: ClipInfo[], onlyPaths = paths) {
  const per: Record<string, any> = {}
  for (const p of onlyPaths) {
    const runs = clips.map((c) => ({ clip: c, r: scoreRun(p, c, 1, root) }))
    if (runs.some((x) => !x.r)) {
      per[p] = null
      continue
    }
    const rs = runs.map((x) => x.r!)
    const ops = rs.reduce((a, r) => a + r.werOps.sub + r.werOps.ins + r.werOps.del, 0)
    const ref = rs.reduce((a, r) => a + r.werOps.ref, 0)
    const heardW = rs.reduce((a, r) => a + r.heard * r.referenceWords, 0)
    const refW = rs.reduce((a, r) => a + r.referenceWords, 0)
    const spans = runs.flatMap(({ clip, r }) =>
      r!.loopSpans.map((l: any) => ({ clip: clip.name, ...l, ...loopKind(clip, l.text) }))
    )
    per[p] = {
      loops: spans.length,
      copyLoops: spans.filter((s) => s.kind === 'copy').length,
      madeUpLoops: spans.filter((s) => s.kind === 'made-up').length,
      clipsWithLoop: rs.filter((r) => r.loops > 0).length,
      loopWords: rs.reduce((a, r) => a + r.loopWords, 0),
      lost: rs.reduce((a, r) => a + r.lost.length, 0),
      heard: heardW / refW,
      wer: ops / ref,
      promptEcho: rs.filter((r) => r.promptEcho).length,
      errors: rs.filter((r) => r.error).length,
      loopSpans: spans,
      clips: runs.map(({ clip, r }) => ({
        clip: clip.name,
        loops: r!.loops,
        loopWords: r!.loopWords,
        lost: r!.lost.map((l: any) => l.clipTime),
        wer: r!.wer,
      })),
    }
  }
  return per
}

function fleurs(id: string) {
  const d = join(FINAL, 'fleurs', id)
  if (!existsSync(d)) return null
  const { per, ...s } = fleursScore(d)
  return s.missing > 0 ? { ...s, incomplete: true } : s
}

function main() {
  const heldout = loadClips('heldout')
  const res: Record<string, any> = {}
  for (const id of IDS) {
    const long = longClips(join(FINAL, 'transcripts', id), heldout)
    const loopsBar = BAR_LOOP_PATHS.every((p) => long[p]) ? BAR_LOOP_PATHS.reduce((a, p) => a + long[p].loops, 0) : null
    res[id] = { id, long, loopsBar, fleurs: fleurs(id) }
  }
  const base = res.baseline
  for (const id of IDS.slice(1)) {
    const r = res[id]
    r.loopDropBar = r.loopsBar === null || base.loopsBar === null ? null : base.loopsBar === 0 ? null : 1 - r.loopsBar / base.loopsBar
  }
  const e3 = res.E3
  const known = (...xs: any[]) => xs.every((x) => x !== null && x !== undefined)
  const bar = {
    c1_loopsDown80: known(e3.loopDropBar) ? e3.loopDropBar >= 0.8 : null,
    c2_controlsNoEffect: CONTROLS.every((c) => known(res[c].loopDropBar)) ? CONTROLS.every((c) => res[c].loopDropBar < 0.8) : null,
    c3_lostNotWorsePerPath: BAR_LOST_PATHS.every((p) => e3.long[p] && base.long[p])
      ? BAR_LOST_PATHS.every((p) => e3.long[p].lost <= base.long[p].lost)
      : null,
    c4_ntVadNoLoops: e3.long['nt-vad'] ? e3.long['nt-vad'].loops === 0 : null,
    c5_fleursWithin03pp:
      e3.fleurs && base.fleurs && !e3.fleurs.incomplete && !base.fleurs.incomplete
        ? e3.fleurs.wer <= base.fleurs.wer + 0.003
        : null,
  }
  const verdict = Object.values(bar).every((v) => v === true) ? 'PASS' : Object.values(bar).some((v) => v === false) ? 'FAIL' : 'INCOMPLETE'

  // Extras (report only).
  const extras: Record<string, any> = {}
  const eng = join(FINAL, 'extras/english')
  if (existsSync(join(REPO, 'data/manifests/english.json')) && existsSync(eng)) {
    const clips = loadClips('english')
    extras.english = Object.fromEntries(
      ['baseline', 'E3'].map((id) => [id, longClips(join(eng, id), clips, ['timestamps', 'nt', 'nt-vad'])])
    )
  }
  const q = join(FINAL, 'extras/q5_0')
  if (existsSync(q)) {
    extras.q5_0 = Object.fromEntries(
      ['baseline', 'E3'].map((id) => [id, longClips(join(q, id), heldout, ['timestamps', 'nt'])])
    )
  }
  const cod = join(FINAL, 'extras/codictate')
  if (existsSync(cod)) {
    const clips = loadClips('codictate')
    const da = clips.filter((c: any) => c.lang === 'da')
    const en = clips.filter((c: any) => c.lang === 'en')
    extras.codictate = Object.fromEntries(
      ['baseline', 'E3'].map((id) => [
        id,
        {
          da: longClips(join(cod, id), da, ['timestamps', 'nt', 'nt-vad']),
          en: longClips(join(cod, id), en, ['timestamps', 'nt', 'nt-vad']),
        },
      ])
    )
  }
  const dict = join(FINAL, 'extras/dictation')
  if (existsSync(dict))
    extras.dictation = Object.fromEntries(['baseline', 'E3'].map((id) => [id, dictationScore(join(dict, id))]))

  mkdirSync(FINAL, { recursive: true })
  writeFileSync(
    join(FINAL, 'results.json'),
    JSON.stringify(
      {
        generated: new Date().toISOString(),
        adr: 'docs/adr/0002-timestamps-bare-leaves-the-bar.md',
        set: 'Held-out Set, 30 Long Clips (data/manifests/heldout.json), run 1',
        controls: CONTROLS,
        bar,
        verdict,
        results: res,
        extras,
      },
      null,
      1
    ) + '\n'
  )

  const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? '-' : (x * 100).toFixed(d))
  console.log('| Model | Loops ts / nt (copy) | ts+nt | drop | bare | silero / vad Loops | lost ts / nt / silero / vad / bare | WER ts / nt / silero / vad / bare | FLEURS test WER |')
  console.log('|---|---|---|---|---|---|---|---|---|')
  for (const id of IDS) {
    const r = res[id]
    const L = r.long
    const c = (p: string, f: (x: any) => string) => (L[p] ? f(L[p]) : '-')
    console.log(
      `| ${id} | ${c('timestamps', (x) => `${x.loops} (${x.copyLoops})`)} / ${c('nt', (x) => `${x.loops} (${x.copyLoops})`)} | ${r.loopsBar ?? '-'} | ${id === 'baseline' ? '' : pct(r.loopDropBar, 0) + '%'} | ${c('timestamps-bare', (x) => `${x.loops} (${x.copyLoops})`)} | ${c('nt-silero', (x) => x.loops)} / ${c('nt-vad', (x) => x.loops)} | ${['timestamps', 'nt', 'nt-silero', 'nt-vad', 'timestamps-bare'].map((p) => c(p, (x) => String(x.lost))).join(' / ')} | ${['timestamps', 'nt', 'nt-silero', 'nt-vad', 'timestamps-bare'].map((p) => c(p, (x) => pct(x.wer))).join(' / ')} | ${r.fleurs ? `${pct(r.fleurs.wer, 2)}%${r.fleurs.incomplete ? ' (partial)' : ''}` : '-'} |`
    )
  }
  console.log(`\nbar: ${JSON.stringify(bar)} -> ${verdict}`)
  for (const id of IDS) {
    for (const p of paths) {
      const L = res[id].long[p]
      if (!L) continue
      for (const s of L.loopSpans)
        console.log(`  ${id} ${p} ${s.clip} [${s.kind} ${s.words}w @${s.clipTime}] ${s.text.slice(0, 160)}`)
    }
  }
}

if (import.meta.main) main()
