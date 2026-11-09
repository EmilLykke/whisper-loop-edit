/**
 * Where Codictate lives, and the few helpers this repo imports from it, read-only.
 *
 * Set CODICTATE_DIR to a checkout of https://github.com/EmilLykke/codictate at commit 0c25733
 * (the commit every result here was produced with). Default: ../codictate next to this repo.
 * It needs Codictate's vendored crispasr (vendors/crispasr/) and silero model (vendors/silero-vad/),
 * and, for FLEURS / LibriSpeech, its benchmark datasets under benchmarks/datasets/.
 */

import { join, resolve } from 'node:path'

export const CODICTATE = resolve(process.env.CODICTATE_DIR ?? resolve(import.meta.dir, '../../codictate'))
export const CODICTATE_COMMIT = '0c25733' // ../codictate HEAD when this ran

const wer = await import(join(CODICTATE, 'benchmarks/stt/wer.ts'))
export const computeWer: (reference: string, hypothesis: string) => any = wer.computeWer
export const tokenizeForWer: (text: string) => string[] = wer.tokenizeForWer

const harness = await import(join(CODICTATE, 'src/bun/utils/whisper/whisper-harness-command.ts'))
export const buildWhisperHarnessCommand = harness.buildWhisperHarnessCommand
export const DICTATION_PROMPT: string = harness.DICTATION_PROMPT

const vad = await import(join(CODICTATE, 'src/bun/utils/whisper/engines/vad-slices.ts'))
export const planVadSlices = vad.planVadSlices
export const vadImportJson = vad.vadImportJson
export const readWavSamples = vad.readWavSamples

const cj = await import(join(CODICTATE, 'src/bun/utils/whisper/engines/crispasr-json.ts'))
export const parseCrispasrJson = cj.parseCrispasrJson
export const crispasrTranscript = cj.crispasrTranscript

const manifests = await import(join(CODICTATE, 'benchmarks/scripts/build-manifests.ts'))
export const fleursEntriesFromTsv = manifests.fleursEntriesFromTsv
