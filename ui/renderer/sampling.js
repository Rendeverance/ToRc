// sampling.js - how a reply's tokens are picked. By default each model gets its maker's recommended sampling, taken
// from its model card: with thinking, without, and for precise coding where the maker gives a preset for it. The
// Replies group on the Load a model sheet chooses between that, coding, greedy and custom values.

// [temperature, top_p, top_k, presence penalty] (top_k 0: no cut; no fourth value: no penalty). The model is told
// apart by its architecture and, for the two that share deepseek2, its file name.
const SAMPLING = [
  {
    is: (a) => a === 'qwen35moe' || a === 'qwen35',
    name: 'Qwen3.5 / 3.6 (Qwen model card)',
    think: [1.0, 0.95, 20, 1.5],
    code: [0.6, 0.95, 20, 0],
    plain: [0.7, 0.8, 20, 1.5],
  },
  {
    is: (a) => a === 'qwen3moe' || a === 'qwen3next' || a === 'qwen4exp',
    name: 'Qwen3 (Qwen model card)',
    think: [0.6, 0.95, 20],
    plain: [0.7, 0.8, 20],
  },
  // Gemma 4 and gpt-oss run on an engine path that is greedy only so far: their makers' settings are kept for when it
  // samples, and the note says so
  {
    is: (a) => a === 'gemma4',
    name: 'Gemma 4 (Google model card; greedy for now: this engine path does not sample yet)',
    think: [1.0, 0.95, 64],
    plain: [1.0, 0.95, 64],
  },
  {
    is: (a) => a === 'gpt-oss',
    name: 'gpt-oss (OpenAI; greedy for now: this engine path does not sample yet)',
    think: [1.0, 1.0, 0],
    plain: [1.0, 1.0, 0],
  },
  {
    is: (a, n) => a === 'deepseek2' && /glm/.test(n),
    name: 'GLM-4.7-Flash (its card’s evaluation settings; top_k 40 ours)',
    think: [1.0, 0.95, 40],
    plain: [1.0, 0.95, 40],
  },
  { is: (a) => a === 'lfm2moe', name: 'LFM2.5 (Liquid AI model card)', think: [0.2, 1.0, 80], plain: [0.2, 1.0, 80] },
];
// a model whose maker publishes no settings (ERNIE 4.5, Moonlight): a common middle ground, said as such
const SAMPLING_NONE = { name: 'no settings published by its maker: a general default', think: [0.7, 0.95, 40] };
SAMPLING_NONE.plain = SAMPLING_NONE.think;

// The entry for a model ({arch, name} from window.torc.modelArch).
function samplingOf(model) {
  const a = (model && model.arch) || '';
  const n = (model && model.name) || '';
  return SAMPLING.find((e) => e.is(a, n)) || SAMPLING_NONE;
}

// What a request sends: { temperature, top_p, top_k } for a turn with or without thinking.
function samplingFor(think) {
  const s = S.settings;
  const mode = s.sampling || 'recommended';
  if (mode === 'greedy') return { temperature: 0, top_p: 1, top_k: 1, presence_penalty: 0, frequency_penalty: 0 };
  if (mode === 'custom') {
    const n = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
    const [presence_penalty, frequency_penalty] = [n(s.sPresence, 0), n(s.sFrequency, 0)];
    const top_k = Math.round(n(s.sTopK, 20));
    return { temperature: n(s.sTemp, 0.7), top_p: n(s.sTopP, 0.95), top_k, presence_penalty, frequency_penalty };
  }
  const e = samplingOf(S.modelArch);
  const v = (mode === 'code' && e.code) || (think ? e.think : e.plain);
  return { temperature: v[0], top_p: v[1], top_k: v[2], presence_penalty: v[3] || 0, frequency_penalty: 0 };
}

// One line saying what Recommended means for the model.
function samplingNote(model) {
  const e = samplingOf(model);
  const pen = (v) => (v[3] ? `, presence penalty ${v[3]}` : '');
  const f = (v) => `temperature ${v[0]}, top_p ${v[1]}, top_k ${v[2] || 'off'}${pen(v)}`;
  const think = f(e.think);
  const plain = e.plain === e.think || f(e.plain) === think ? '' : `; without thinking ${f(e.plain)}`;
  const code = e.code ? `; Coding: ${f(e.code)}` : '';
  return `${e.name}: ${think}${plain}${code}`;
}
