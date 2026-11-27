const KINDS = {
  kick:  { name: "킥",     hint: "'붐', '둠' 같은 낮은 소리",   color: "#ff6b4a" },
  snare: { name: "스네어", hint: "'카', '프슈' 같은 터지는 소리", color: "#ffd23f" },
  hihat: { name: "하이햇", hint: "'츠', '칫' 같은 짧은 소리",    color: "#4ad6ff" },
};
const CAL_TARGET = 8;          // 종류마다 학습할 횟수
const CAL_MAX_SECONDS = 25;
const MAX_SECONDS = 30;
const STORE_KEY = "mouthdrums-model-v1";
const $ = (id) => document.getElementById(id);

let model = loadModel();
let take = null;            // { x, sr } 마지막 녹음
let recorder = null;
let urls = [];

function loadModel() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)); } catch { return null; }
}
function saveModel() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(model)); } catch {}
}

// ---------- 마이크 ----------
async function startRecording(onLevel) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("이 브라우저에서는 마이크를 쓸 수 없습니다. (https 주소인지 확인해 주세요)");
  }
  // 통화용 음성 처리는 타악기 소리를 뭉개므로 모두 끈다
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const chunks = [];
  const onChunk = (c) => {
    chunks.push(c);
    let p = 0;
    for (let i = 0; i < c.length; i++) p = Math.max(p, Math.abs(c[i]));
    onLevel(p);
  };
  let node;
  if (ctx.audioWorklet) {
    await ctx.audioWorklet.addModule("recorder-worklet.js");
    node = new AudioWorkletNode(ctx, "rec");
    node.port.onmessage = (e) => onChunk(e.data);
  } else {
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
  }
  const mute = ctx.createGain();
  mute.gain.value = 0;
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  return {
    sr: ctx.sampleRate,
    snapshot() {                                      // 녹음을 멈추지 않고 지금까지의 소리를 본다
      const x = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
      let o = 0;
      for (const c of chunks) { x.set(c, o); o += c.length; }
      return x;
    },
    stop() {
      stream.getTracks().forEach((t) => t.stop());   // 마이크를 놓아야 재생 음량이 정상으로 돌아온다
      node.disconnect();
      ctx.close();
      const x = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
      let o = 0;
      for (const c of chunks) { x.set(c, o); o += c.length; }
      return x;
    },
  };
}

function micError(e) {
  if (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) {
    return "마이크 권한이 필요합니다. 브라우저 설정에서 마이크를 허용해 주세요.";
  }
  return (e && e.message) || "마이크를 열 수 없습니다.";
}

// ---------- 1. 학습 ----------
function renderCalList() {
  $("calList").innerHTML = "";
  for (const label of Engine.LABELS) {
    const k = KINDS[label];
    const n = model && model.examples[label] ? model.examples[label].length : 0;
    const row = document.createElement("div");
    row.className = "cal";
    const stateClass = n >= 4 ? "ok" : n > 0 ? "warn" : "";
    const stateText = n >= 4 ? `${n}개 학습됨` : n > 0 ? `${n}개뿐이에요. 다시 녹음해 주세요` : k.hint;
    row.innerHTML = `<span class="dot" style="background:${k.color}"></span>
      <div class="info"><div class="name">${k.name}</div><div class="state ${stateClass}">${stateText}</div></div>
      <button>${n ? "다시 녹음" : "녹음"}</button>`;
    row.querySelector("button").onclick = () => calibrate(label);
    $("calList").appendChild(row);
  }
  const ready = Engine.isReady(model);
  $("recBtn").disabled = !ready;
  if (!recorder) $("recInfo").textContent = ready ? "버튼을 누르고 비트를 불러보세요." : "세 가지 소리를 먼저 학습시켜 주세요.";
}

// 녹음 중에 지금까지 인식한 횟수를 센다. 매번 전체를 다시 분석하면 휴대폰에서 버벅이므로,
// 지나간 부분은 횟수만 확정해 두고 최근 몇 초만 다시 본다.
function liveCounter(sr) {
  let base = 0, committed = 0, maxLevel = 0;
  return (x) => {
    const tail = x.subarray(base);
    const r = Engine.calibrationOnsets(tail, sr, maxLevel);
    maxLevel = r.maxLevel;
    const total = committed + r.onsets.length;
    const cut = tail.length - sr;
    // 소리 한가운데를 자르면 두 번 세거나 놓치므로, 조용한 지점일 때만 확정한다
    if (tail.length > 3 * sr && !r.onsets.some((o) => o > cut - 0.3 * sr && o < cut + 0.1 * sr)) {
      committed += r.onsets.filter((o) => o < cut).length;
      base += cut;
    }
    return total;
  };
}

async function calibrate(label) {
  if (recorder) return;
  const k = KINDS[label];
  $("msg").textContent = "";
  let rec;
  try {
    rec = await startRecording((p) => ($("ovMeter").style.width = Math.min(100, p * 140) + "%"));
  } catch (e) { $("msg").textContent = micError(e); return; }
  recorder = rec;
  $("ovTitle").textContent = `${k.name} 소리를 내주세요`;
  $("ovHint").textContent = `${k.hint} · 소리를 낼 때마다 숫자가 줄어듭니다`;
  $("ovCount").textContent = CAL_TARGET;
  $("ovCount").style.color = k.color;
  $("overlay").classList.add("on");

  // 인식한 횟수만큼 숫자를 줄이고, 다 채우면 자동으로 끝낸다
  let stopEarly = false, shown = CAL_TARGET;
  const count = liveCounter(rec.sr);
  $("ovStop").onclick = () => (stopEarly = true);
  const started = Date.now();
  while (!stopEarly && Date.now() - started < CAL_MAX_SECONDS * 1000) {
    await new Promise((r) => setTimeout(r, 150));
    const left = Math.max(0, CAL_TARGET - count(rec.snapshot()));
    if (left !== shown) {
      shown = left;
      $("ovCount").textContent = left;
      $("ovCount").animate([{ transform: "scale(1.35)" }, { transform: "scale(1)" }], 180);
    }
    if (left === 0) {                                  // 마지막 소리의 꼬리까지 담고 끝낸다
      await new Promise((r) => setTimeout(r, 350));
      break;
    }
  }
  const x = rec.stop();
  recorder = null;
  $("overlay").classList.remove("on");

  if (!model || model.sr !== rec.sr) model = Engine.newModel(rec.sr);
  const found = Engine.learn(model, label, x).length;
  saveModel();
  renderCalList();
  if (found === 0) $("msg").textContent = "소리를 찾지 못했습니다. 조금 더 크게, 가까이서 내주세요.";
  if (take) convert();
}

// ---------- 2. 녹음 / 변환 ----------
async function toggleRecord() {
  if (recorder) return finishRecord();
  $("msg").textContent = "";
  const meter = $("recMeter").firstElementChild;
  try {
    recorder = await startRecording((p) => (meter.style.width = Math.min(100, p * 140) + "%"));
  } catch (e) { $("msg").textContent = micError(e); return; }
  $("recBtn").classList.add("on");
  $("recBtn").textContent = "멈춤";
  const started = Date.now();
  recorder.timer = setInterval(() => {
    const s = (Date.now() - started) / 1000;
    $("recInfo").textContent = `녹음 중 ${s.toFixed(1)}초`;
    if (s >= MAX_SECONDS) finishRecord();
  }, 100);
}

function finishRecord() {
  const rec = recorder;
  if (!rec) return;
  recorder = null;
  clearInterval(rec.timer);
  $("recBtn").classList.remove("on");
  $("recBtn").textContent = "녹음";
  $("recMeter").firstElementChild.style.width = "0";
  take = { x: rec.stop(), sr: rec.sr };
  convert();
}

function convert() {
  if (!take || !Engine.isReady(model)) return;
  if (take.sr !== model.sr) {
    $("msg").textContent = "녹음 환경이 학습 때와 달라졌습니다. 세 가지 소리를 다시 학습시켜 주세요.";
    return;
  }
  const found = Engine.transcribe(take.x, model, parseFloat($("sens").value));
  const grid = $("quant").checked ? Engine.quantize(found, take.sr) : null;
  const division = parseInt($("hats").value, 10);
  $("hats").disabled = !grid;                         // 박자를 알아야 하이햇을 깔 수 있다
  const hits = !grid ? found : division ? Engine.addHats(grid, division, take.sr) : grid.hits;
  // 박자를 맞췄으면 MIDI는 첫 소리가 첫 박에 오도록 해서 DAW 격자와 바로 맞게 한다
  const midiHits = grid ? hits.map((h) => ({ ...h, time: h.time - hits[0].time })) : hits;
  const drums = Engine.render(hits, take.x.length, take.sr);

  urls.forEach(URL.revokeObjectURL);
  const url = (data, type) => { const u = URL.createObjectURL(new Blob([data], { type })); urls.push(u); return u; };
  const drumUrl = url(Engine.encodeWav(drums, take.sr), "audio/wav");
  $("drumAudio").src = drumUrl;
  $("voiceAudio").src = url(Engine.encodeWav(take.x, take.sr), "audio/wav");
  $("dlWav").href = drumUrl;
  $("dlMid").href = url(Engine.encodeMidi(midiHits, grid ? Math.round(grid.bpm) : 120), "audio/midi");

  $("result").style.display = "block";
  const counts = {};
  hits.forEach((h) => (counts[h.label] = (counts[h.label] || 0) + 1));
  $("legend").innerHTML = Engine.LABELS.map((l) =>
    `<span><i class="dot" style="background:${KINDS[l].color}"></i>${KINDS[l].name} ${counts[l] || 0}</span>`).join("");
  $("recInfo").textContent = hits.length ? `${hits.length}개 소리를 찾았습니다.` : "소리를 찾지 못했습니다. 민감도를 올려보세요.";
  $("quantInfo").textContent = !$("quant").checked ? "" : grid ? `약 ${Math.round(grid.bpm)} BPM` : "박자를 찾지 못해 그대로 뒀어요";
  draw(take.x, found);                    // 파형 위에는 실제로 소리가 난 위치를 표시
}

function draw(x, hits) {
  const cv = $("view"), dpr = window.devicePixelRatio || 1;
  const w = (cv.width = Math.round(cv.clientWidth * dpr)), h = (cv.height = Math.round(cv.clientHeight * dpr));
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  g.fillStyle = "#4a4a60";
  const step = x.length / w;
  for (let px = 0; px < w; px++) {
    let p = 0;
    for (let i = Math.floor(px * step), e = Math.floor((px + 1) * step); i < e; i++) p = Math.max(p, Math.abs(x[i]));
    const bar = Math.max(1, p * h * 0.9);
    g.fillRect(px, (h - bar) / 2, 1, bar);
  }
  for (const hit of hits) {
    g.fillStyle = KINDS[hit.label].color;
    g.fillRect(Math.round((hit.sample / x.length) * w), 0, Math.max(2, 2 * dpr), h);
  }
}

$("recBtn").onclick = toggleRecord;
$("quant").onchange = convert;
$("hats").onchange = convert;
// 민감도는 한 번 맞춰두면 다음에도 그대로 쓰도록 기억한다
const SENS_KEY = "mouthdrums-sens-v2";
const savedSens = parseFloat(localStorage.getItem(SENS_KEY));
if (savedSens >= 0.3 && savedSens <= 1.3) $("sens").value = savedSens;
$("sens").oninput = () => {
  try { localStorage.setItem(SENS_KEY, $("sens").value); } catch {}
  convert();
};
window.addEventListener("resize", () => take && convert());
renderCalList();
