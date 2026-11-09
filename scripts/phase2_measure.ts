/**
 * Phase 2: score every edit against the baseline -> results/phase2/edits.json, and print the
 * summary table as markdown.
 *
 *   bun scripts/phase2_measure.ts                                   Phase 2: results/phase2/edits.json
 *   bun scripts/phase2_measure.ts --out results/phase2b/edits.json baseline E3 C1 C2 E7 E9 C3
 *
 * Long clips: Phase 0's scoring unchanged (`scoreRun` from measure.ts: the broader Loop detector,
 * heard / lost utterances by `heardPerWindow`, WER by Codictate's `computeWer`), on
 * results/phase2/transcripts/<id>/<path>/ run 1; the baseline is Phase 0's own run 1 (same
 * harness, argv, crispasr build and thread count; its ggml is tensor-identical to the official
 * file). Pooled WER = all clips' edit operations / all reference words; pooled heard = heard
 * reference words / reference words.
 * Spot check: run 2 of (timestamps, find_da_5min_04) must give byte-identical JSON to run 1.
 * Short utterances: results/phase2/transcripts/<id>/utts/, pooled WER over the 155 Find Set
 * utterances (computeWer per utterance, operations summed).
 * HF proxy: results/phase2/proxy/<id>.json (phase2_proxy.py).
 *
 * Pre-registered bar (flagged, not ranked): Loops on timestamps + timestamps-bare + nt down by
 * >= 80% against baseline; Lost Audio not worse (lost utterances summed over the five paths <=
 * baseline; per path reported too); short-utterance WER <= baseline + 0.3 pp; nt-vad 0 Loops; and
 * the matching Control Edit does not show the effect (its own Loop drop on the three paths < 80%).
 * Each Loop is also tagged by kind (rough): `copy` when at least 10% of its word trigrams occur in the
 * clip's reference (it writes heard speech again, often misheard; Phase 0's kind), else `made-up` (it repeats text
 * the audio never says, such as subtitle credits or filler).
 * Controls: C1 (11 random heads) for the head edits E1, E2, E3; C2 (random direction, a = 2) for the
 * direction edits E4, E5; both for E6. Phase 2b: C1 + C2 for E7 and E8 (E3 + direction, a = 1 / 0.5);
 * C3 (as many random heads per layer as E9) for E9 (E3 + Calm-Whisper heads #1, #6, #11 in all layers).
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { computeWer, tokenizeForWer } from './codictate'
import { scoreRun } from './measure'
import { findUtterances } from './phase2_utts'
import { findClips, OUT as P0_OUT, PATHS, REPO } from './run_paths'

const P2 = join(REPO, 'results/phase2')
const argv = process.argv.slice(2)
const outAt = argv.indexOf('--out')
const OUT_JSON = outAt >= 0 ? join(REPO, argv[outAt + 1]) : join(P2, 'edits.json')
const argIds = argv.filter((_, i) => i !== outAt && i !== outAt + 1)
const IDS = argIds.length ? argIds : ['baseline', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'C1', 'C2']
const CONTROLS: Record<string, string[]> = {
  E1: ['C1'], E2: ['C1'], E3: ['C1'], E4: ['C2'], E5: ['C2'], E6: ['C1', 'C2'],
  E7: ['C1', 'C2'], E8: ['C1', 'C2'], E9: ['C3'],
}
const LOOP_PATHS = ['timestamps', 'timestamps-bare', 'nt']
const paths = Object.keys(PATHS)
const clips = findClips()
const utts = findUtterances()

function loopKind(clipName: string, text: string): { refTrigramShare: number; kind: 'copy' | 'made-up' } {
  const clip = clips.find((c) => c.name === clipName)!
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

function longClips(id: string) {
  const root = id === 'baseline' ? P0_OUT : join(P2, 'transcripts', id)
  const per: Record<string, any> = {}
  for (const p of paths) {
    const runs = clips.map((c) => ({ clip: c.name, r: scoreRun(p, c, 1, root) }))
    if (runs.some((x) => !x.r)) return null
    const rs = runs.map((x) => x.r!)
    const ops = rs.reduce((a, r) => a + r.werOps.sub + r.werOps.ins + r.werOps.del, 0)
    const ref = rs.reduce((a, r) => a + r.werOps.ref, 0)
    const heardW = rs.reduce((a, r) => a + r.heard * r.referenceWords, 0)
    const refW = rs.reduce((a, r) => a + r.referenceWords, 0)
    per[p] = {
      loops: rs.reduce((a, r) => a + r.loops, 0),
      copyLoops: rs.reduce((a, r, i) => a + r.loopSpans.filter((l: any) => loopKind(runs[i].clip, l.text).kind === 'copy').length, 0),
      clipsWithLoop: rs.filter((r) => r.loops > 0).length,
      loopWords: rs.reduce((a, r) => a + r.loopWords, 0),
      repeatedParts: rs.reduce((a, r) => a + r.repeatedParts, 0),
      lost: rs.reduce((a, r) => a + r.lost.length, 0),
      heard: heardW / refW,
      wer: ops / ref,
      promptEcho: rs.filter((r) => r.promptEcho).length,
      errors: rs.filter((r) => r.error).length,
      clips: runs.map(({ clip, r }) => ({
        clip,
        loops: r!.loops,
        loopWords: r!.loopWords,
        heard: r!.heard,
        lost: r!.lost.map((l: any) => l.clipTime),
        wer: r!.wer,
        loopSpans: r!.loopSpans.map((l: any) => ({ ...l, ...loopKind(clip, l.text) })),
        promptEcho: r!.promptEcho,
      })),
    }
  }
  return per
}

function spotCheck(id: string) {
  if (id === 'baseline') return { identical: true, note: 'Phase 0: 50/50 pairs byte-identical' }
  const d = join(P2, 'transcripts', id, 'timestamps')
  const a = join(d, 'find_da_5min_04.run1.json')
  const b = join(d, 'find_da_5min_04.run2.json')
  if (!existsSync(a) || !existsSync(b)) return null
  return { identical: readFileSync(a).equals(readFileSync(b)), clip: 'find_da_5min_04', path: 'timestamps' }
}

function shortUtts(id: string) {
  const dir = join(P2, 'transcripts', id, 'utts')
  let ops = 0
  let ref = 0
  const rows: { name: string; wer: number; ops: number; ref: number; text: string }[] = []
  for (const u of utts) {
    const f = join(dir, `${u.name}.txt`)
    if (!existsSync(join(dir, `${u.name}.meta.json`))) return null
    const text = readFileSync(f, 'utf8').trim()
    const w = computeWer(u.text, text)
    const o = w.substitutions + w.insertions + w.deletions
    ops += o
    ref += w.refWords
    rows.push({ name: u.name, wer: w.wer, ops: o, ref: w.refWords, text })
  }
  return { n: rows.length, wer: ops / ref, ops, ref, empty: rows.filter((r) => !r.text).length, rows }
}

function proxy(id: string) {
  const f = join(P2, 'proxy', `${id}.json`)
  if (!existsSync(f)) return null
  const r = JSON.parse(readFileSync(f, 'utf8'))
  return {
    loopRate: r.loop_rate,
    fuzzyLoopRate: r.fuzzy_loop_rate,
    loopLd: r.loop_ld,
    cleanContLp: r.clean_cont_lp,
    cleanUnchanged: r.clean_greedy_unchanged,
    natural: Object.values(r.natural.per_case).filter(Boolean).length,
    naturalPerCase: r.natural.per_case,
    naturalGens: r.natural.gens,
    naturalCleanContLp: r.natural.clean_cont_lp,
  }
}

function main() {
  const res: Record<string, any> = {}
  for (const id of IDS) {
    const lc = longClips(id)
    const su = shortUtts(id)
    res[id] = { id, proxy: proxy(id), long: lc, spotCheck: lc ? spotCheck(id) : null, short: su }
    if (lc) {
      let worst: any = null
      for (const p of paths)
        for (const c of lc[p].clips)
          for (const s of c.loopSpans)
            if (!worst || s.words > worst.words) worst = { path: p, clip: c.clip, ...s }
      res[id].worstLoop = worst
      res[id].loops3 = LOOP_PATHS.reduce((a, p) => a + lc[p].loops, 0)
      res[id].lostAll = paths.reduce((a, p) => a + lc[p].lost, 0)
    }
  }
  const base = res.baseline
  const shortRows = new Map<string, any>((base.short?.rows ?? []).map((r: any) => [r.name, r]))
  for (const id of IDS) {
    const r = res[id]
    if (id === 'baseline' || !r.long || !r.short || !base.short) continue
    r.loopDrop3 = 1 - r.loops3 / base.loops3
    r.shortChanged = r.short.rows
      .filter((x: any) => x.text !== shortRows.get(x.name)?.text)
      .map((x: any) => ({ name: x.name, ops: x.ops, baseOps: shortRows.get(x.name).ops, text: x.text, baseText: shortRows.get(x.name).text }))
  }
  for (const id of IDS) {
    const r = res[id]
    if (id === 'baseline' || r.loopDrop3 === undefined) continue
    const ctrls = (CONTROLS[id] ?? []).map((c) => res[c])
    const bar: Record<string, boolean | null> = {
      loopsDown80: r.loopDrop3 >= 0.8,
      lostNotWorse: r.lostAll <= base.lostAll,
      lostNotWorsePerPath: paths.every((p) => r.long[p].lost <= base.long[p].lost),
      shortWerWithin03pp: r.short.wer <= base.short.wer + 0.003,
      ntVadNoLoops: r.long['nt-vad'].loops === 0,
      controlNoEffect: ctrls.length === 0 ? null : ctrls.every((c) => c.loopDrop3 !== undefined) ? ctrls.every((c) => c.loopDrop3 < 0.8) : null,
    }
    r.bar = bar
    r.meetsBar = !id.startsWith('C') && Object.entries(bar).filter(([k]) => k !== 'lostNotWorsePerPath').every(([, v]) => v === true)
  }
  mkdirSync(dirname(OUT_JSON), { recursive: true })
  writeFileSync(
    OUT_JSON,
    JSON.stringify(
      {
        generated: new Date().toISOString(),
        baselineLongClips: 'results/phase0/transcripts run 1 (Phase 0, same harness)',
        controls: CONTROLS,
        bar: 'Loops on timestamps+timestamps-bare+nt down >= 80%; lost utterances over the 5 paths <= baseline; short-utterance WER <= baseline + 0.3 pp; nt-vad 0 Loops; control Loop drop < 80%',
        results: res,
      },
      null,
      1
    ) + '\n'
  )

  const pct = (x: number | undefined | null, d = 1) => (x === undefined || x === null ? '-' : (x * 100).toFixed(d))
  console.log('| Edit | find-B Loop | clean log-p | natural | Loops ts / bare / nt / silero / vad | ts heard / lost | nt heard / lost | long WER ts / bare / nt / silero / vad | short WER | nt-vad lost | bar |')
  console.log('|---|---|---|---|---|---|---|---|---|---|---|')
  for (const id of IDS) {
    const r = res[id]
    const px = r.proxy
    const lc = r.long
    const cells = [
      id,
      px ? `${pct(px.loopRate, 0)}%` : '-',
      px ? px.cleanContLp.toFixed(3) : '-',
      px ? `${px.natural}/5` : '-',
      lc ? paths.map((p) => lc[p].loops).join(' / ') : '-',
      lc ? `${pct(lc.timestamps.heard)}% / ${lc.timestamps.lost}` : '-',
      lc ? `${pct(lc.nt.heard)}% / ${lc.nt.lost}` : '-',
      lc ? paths.map((p) => pct(lc[p].wer)).join(' / ') : '-',
      r.short ? `${pct(r.short.wer, 2)}%` : '-',
      lc ? String(lc['nt-vad'].lost) : '-',
      r.bar ? (r.meetsBar ? 'MEETS' : 'no: ' + Object.entries(r.bar).filter(([k, v]) => v === false && k !== 'lostNotWorsePerPath').map(([k]) => k).join(', ')) : '-',
    ]
    console.log(`| ${cells.join(' | ')} |`)
  }
  console.log()
  for (const id of IDS) {
    const r = res[id]
    if (!r.long) continue
    const w = r.worstLoop
    console.log(`${id}: copy loops ${paths.map((p) => r.long[p].copyLoops).join('/')} | spot ${r.spotCheck?.identical} | loops3 ${r.loops3} drop ${pct(r.loopDrop3, 0)}% | lost all ${r.lostAll} per path ${paths.map((p) => r.long[p].lost).join('/')} | heard ${paths.map((p) => pct(r.long[p].heard)).join('/')} | echo ${paths.map((p) => r.long[p].promptEcho).join('/')} | short changed ${r.shortChanged?.length ?? '-'}`)
    if (w) console.log(`   worst: ${w.path} ${w.clip} ${w.words}w @${w.clipTime}: ${w.text.slice(0, 220)}`)
    if (r.bar) console.log(`   bar: ${JSON.stringify(r.bar)}`)
  }
}

main()
