'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';

type TransformKind = 'clean' | 'jpeg' | 'blur' | 'resize' | 'noise' | 'color' | 'crop';
type Phase = 'idle' | 'analyzing' | 'scanning' | 'done' | 'error';
type BackendState = 'checking' | 'ready' | 'offline';

type TransformSpec = {
  id: string;
  label: string;
  short: string;
  kind: TransformKind;
  value: number;
  realWorld: string;
};

type Prediction = {
  raw_logit: number;
  calibrated_logit: number;
  probability_fake: number;
  aigc_confidence: number;
  label_at_display_threshold: 'aigc' | 'real';
};

type TransformResult = {
  prediction: Prediction;
  transform: { id: string; label: string; operation: string; family?: string; value?: number };
  image: { sha256: string; width: number; height: number };
  branches_available?: string[];
};

type PredictResponse = TransformResult & {
  transform_scan: {
    scan_id: string;
    status: string;
    status_url: string;
    completed_count: number;
    total_count: number;
  };
};

type TransformScan = {
  status: 'queued' | 'running' | 'completed' | 'failed';
  completed_count: number;
  total_count: number;
  current_transform: string | null;
  error: string | null;
  results: Record<string, TransformResult>;
};

type AnalysisResult = {
  explanation: {
    branches?: Array<{ component: string; confidence_delta: number }>;
  };
  result: {
    assets: {
      attribution_overlay: string;
      frequency_overlay: string;
    };
  };
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

const defaultTransforms: TransformSpec[] = [
  { id: 'clean', label: 'Clean', short: 'Clean', kind: 'clean', value: 0, realWorld: 'Original upload' },
  { id: 'jpeg_q90', label: 'JPEG · Q90', short: 'J90', kind: 'jpeg', value: 0.9, realWorld: 'Light social re-encode' },
  { id: 'jpeg_q70', label: 'JPEG · Q70', short: 'J70', kind: 'jpeg', value: 0.7, realWorld: 'Messaging compression' },
  { id: 'jpeg_q50', label: 'JPEG · Q50', short: 'J50', kind: 'jpeg', value: 0.5, realWorld: 'Repeated reposting' },
  { id: 'jpeg_q30', label: 'JPEG · Q30', short: 'J30', kind: 'jpeg', value: 0.3, realWorld: 'Heavy platform compression' },
  { id: 'blur_sigma0.5', label: 'Blur · σ0.5', short: 'B.5', kind: 'blur', value: 0.5, realWorld: 'Mild defocus' },
  { id: 'blur_sigma1', label: 'Blur · σ1.0', short: 'B1', kind: 'blur', value: 1, realWorld: 'Camera or edit blur' },
  { id: 'blur_sigma2', label: 'Blur · σ2.0', short: 'B2', kind: 'blur', value: 2, realWorld: 'Strong defocus' },
  { id: 'resize_0.5x', label: 'Resize · 0.5×', short: 'R.5', kind: 'resize', value: 0.5, realWorld: 'Thumbnail generation' },
  { id: 'resize_0.25x', label: 'Resize · 0.25×', short: 'R.25', kind: 'resize', value: 0.25, realWorld: 'Aggressive thumbnailing' },
  { id: 'noise_sigma0.02', label: 'Noise · σ0.02', short: 'N.02', kind: 'noise', value: 0.02, realWorld: 'Low-light sensor noise' },
  { id: 'noise_sigma0.05', label: 'Noise · σ0.05', short: 'N.05', kind: 'noise', value: 0.05, realWorld: 'Strong sensor noise' },
  { id: 'noise_sigma0.10', label: 'Noise · σ0.10', short: 'N.10', kind: 'noise', value: 0.1, realWorld: 'Extreme sensor noise' },
  { id: 'color_minus20', label: 'Color · −20%', short: 'C−', kind: 'color', value: -0.2, realWorld: 'Muted filter app' },
  { id: 'color_plus20', label: 'Color · +20%', short: 'C+', kind: 'color', value: 0.2, realWorld: 'Auto-enhance filter' },
  { id: 'center_crop_80', label: 'Center crop · 80%', short: 'Crop', kind: 'crop', value: 0.8, realWorld: 'Profile crop or reframing' },
];

const branchLabels: Record<string, string> = {
  global_spatial_and_wavelet_view: 'Global spatial + wavelet',
  semantic_view: 'Semantic view',
  native_tiles: 'Native tiles',
};

const teamMembers = [
  { name: 'Steven Cai', image: '/team/steven-cai.png' },
  { name: 'Xiyan Huang', image: '/team/xiyan-huang.jpg' },
  { name: 'Wenqing Yan', image: '/team/wenqing-yan.jpg' },
  { name: 'Yijun Li', image: '/team/yijun-li.jpg' },
  { name: 'Mingjun Mao', image: '/team/mingjun-mao.jpg' },
];

const wait = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

function apiUrl(pathOrUrl: string) {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  return `${API_BASE}${pathOrUrl.startsWith('/') ? '' : '/'}${pathOrUrl}`;
}

async function responseError(response: Response) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

function seededNoise(x: number, y: number) {
  const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return n - Math.floor(n);
}

function drawContain(ctx: CanvasRenderingContext2D, image: HTMLImageElement, width: number, height: number) {
  const ratio = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const w = image.naturalWidth * ratio;
  const h = image.naturalHeight * ratio;
  ctx.drawImage(image, (width - w) / 2, (height - h) / 2, w, h);
}

function applyTransform(canvas: HTMLCanvasElement, image: HTMLImageElement, spec: TransformSpec) {
  const width = 960;
  const height = 620;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  ctx.fillStyle = '#101010';
  ctx.fillRect(0, 0, width, height);
  if (spec.kind === 'crop') {
    const sw = image.naturalWidth * spec.value;
    const sh = image.naturalHeight * spec.value;
    ctx.drawImage(image, (image.naturalWidth - sw) / 2, (image.naturalHeight - sh) / 2, sw, sh, 0, 0, width, height);
  } else if (spec.kind === 'resize') {
    const temp = document.createElement('canvas');
    temp.width = Math.max(1, Math.round(width * spec.value));
    temp.height = Math.max(1, Math.round(height * spec.value));
    const tempContext = temp.getContext('2d');
    if (tempContext) {
      tempContext.imageSmoothingQuality = 'low';
      drawContain(tempContext, image, temp.width, temp.height);
      ctx.imageSmoothingQuality = 'low';
      ctx.drawImage(temp, 0, 0, width, height);
    }
  } else {
    const blur = spec.kind === 'blur' ? spec.value * 2 : 0;
    const saturation = spec.kind === 'color' ? 1 + spec.value : 1;
    const contrast = spec.kind === 'color' ? 1 + spec.value * 0.65 : 1;
    const brightness = spec.kind === 'color' ? 1 + spec.value * 0.35 : 1;
    ctx.filter = `blur(${blur}px) saturate(${saturation}) contrast(${contrast}) brightness(${brightness})`;
    drawContain(ctx, image, width, height);
    ctx.filter = 'none';
  }
  if (spec.kind === 'jpeg') {
    const encoded = canvas.toDataURL('image/jpeg', spec.value);
    const jpeg = new Image();
    jpeg.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(jpeg, 0, 0, width, height);
    };
    jpeg.src = encoded;
  }
  if (spec.kind === 'noise') {
    const pixels = ctx.getImageData(0, 0, width, height);
    const amplitude = spec.value * 255 * 2.15;
    for (let index = 0; index < pixels.data.length; index += 4) {
      const pixel = index / 4;
      const delta = (seededNoise(pixel % width, Math.floor(pixel / width)) - 0.5) * amplitude;
      for (let channel = 0; channel < 3; channel += 1) {
        pixels.data[index + channel] = Math.max(0, Math.min(255, pixels.data[index + channel] + delta));
      }
    }
    ctx.putImageData(pixels, 0, 0);
  }
}

export default function Home() {
  const [catalog, setCatalog] = useState(defaultTransforms);
  const [selectedId, setSelectedId] = useState('clean');
  const [sourceUrl, setSourceUrl] = useState('/demo-sample.svg');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState('techjam_demo_sample.svg');
  const [phase, setPhase] = useState<Phase>('idle');
  const [backendState, setBackendState] = useState<BackendState>('checking');
  const [error, setError] = useState('');
  const [results, setResults] = useState<Record<string, TransformResult>>({});
  const [scanProgress, setScanProgress] = useState({ completed: 0, total: 16, status: 'idle' });
  const [evidenceView, setEvidenceView] = useState<'attribution' | 'frequency'>('attribution');
  const [evidenceBusy, setEvidenceBusy] = useState(false);
  const [evidenceAssets, setEvidenceAssets] = useState<{ attribution: string; frequency: string } | null>(null);
  const [branchValues, setBranchValues] = useState<Array<{ label: string; value: number }>>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const sourceCanvas = useRef<HTMLCanvasElement>(null);
  const evidenceCanvas = useRef<HTMLCanvasElement>(null);
  const scanToken = useRef(0);
  const analysisToken = useRef(0);

  const active = catalog.find((item) => item.id === selectedId) || catalog[0];
  const currentResult = results[selectedId];
  const confidence = currentResult?.prediction.aigc_confidence;
  const probabilityFake = currentResult?.prediction.probability_fake;
  const rawLogit = currentResult?.prediction.raw_logit;
  const rawLogitText = rawLogit === undefined
    ? '—'
    : `${rawLogit >= 0 ? '+' : ''}${rawLogit.toFixed(4)}`;
  const uncertain = confidence !== undefined && confidence >= 0.42 && confidence <= 0.58;

  useEffect(() => {
    let cancelled = false;
    async function connectBackend() {
      try {
        const health = await fetch(apiUrl('/api/health'));
        if (!health.ok) throw new Error(await responseError(health));
        const response = await fetch(apiUrl('/api/v1/transforms'));
        if (!response.ok) throw new Error(await responseError(response));
        const body = await response.json() as { transforms: Array<{ id: string; label: string }> };
        if (cancelled) return;
        const metadata = new Map(defaultTransforms.map((item) => [item.id, item]));
        setCatalog(body.transforms.map((item) => metadata.get(item.id) || {
          id: item.id,
          label: item.label,
          short: item.id,
          kind: 'clean' as TransformKind,
          value: 0,
          realWorld: item.label,
        }));
        setBackendState('ready');
      } catch {
        if (!cancelled) setBackendState('offline');
      }
    }
    void connectBackend();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      if (sourceCanvas.current) applyTransform(sourceCanvas.current, image, active);
      if (evidenceCanvas.current) applyTransform(evidenceCanvas.current, image, active);
    };
    image.onerror = () => setError('The image could not be decoded. Use a JPG, PNG or WEBP file.');
    image.src = sourceUrl;
  }, [sourceUrl, active]);

  useEffect(() => () => {
    if (sourceUrl.startsWith('blob:')) URL.revokeObjectURL(sourceUrl);
  }, [sourceUrl]);

  async function pollTransformScan(statusUrl: string, token: number) {
    for (let attempt = 0; attempt < 900; attempt += 1) {
      await wait(800);
      if (scanToken.current !== token) return;
      const response = await fetch(apiUrl(statusUrl));
      if (!response.ok) throw new Error(await responseError(response));
      const scan = await response.json() as TransformScan;
      setResults((previous) => ({ ...previous, ...scan.results }));
      setScanProgress({ completed: scan.completed_count, total: scan.total_count, status: scan.status });
      if (scan.status === 'completed') {
        setPhase('done');
        return;
      }
      if (scan.status === 'failed') throw new Error(scan.error || 'Transform scan failed');
      setPhase('scanning');
    }
    throw new Error('Transform scan timed out. The selected result is still available.');
  }

  async function runPrediction(file: File, transformId: string) {
    const token = ++scanToken.current;
    setError('');
    setPhase('analyzing');
    setScanProgress({ completed: 0, total: 16, status: 'starting' });
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('transform', transformId);
      const response = await fetch(apiUrl('/api/v1/predict'), { method: 'POST', body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const body = await response.json() as PredictResponse;
      if (scanToken.current !== token) return;
      setBackendState('ready');
      setSelectedId(body.transform.id);
      setResults((previous) => ({ ...previous, [body.transform.id]: body }));
      setScanProgress({ completed: body.transform_scan.completed_count, total: body.transform_scan.total_count, status: body.transform_scan.status });
      setPhase('scanning');
      await pollTransformScan(body.transform_scan.status_url, token);
    } catch (caught) {
      if (scanToken.current !== token) return;
      const message = caught instanceof Error ? caught.message : 'Prediction failed';
      setError(message === 'Failed to fetch' ? `Cannot reach the inference API at ${API_BASE}. Start the backend, then retry.` : message);
      setPhase('error');
      setBackendState(message === 'Failed to fetch' ? 'offline' : 'ready');
    }
  }

  function selectTransform(transformId: string) {
    setSelectedId(transformId);
    setError('');
    if (results[transformId]) return;
    if (selectedFile) void runPrediction(selectedFile, transformId);
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const supported = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name);
    if (!supported) {
      setError('Please choose a JPG, PNG or WEBP image.');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError('The backend accepts images up to 20 MiB.');
      return;
    }
    scanToken.current += 1;
    analysisToken.current += 1;
    setResults({});
    setEvidenceAssets(null);
    setBranchValues([]);
    setSelectedFile(file);
    setFileName(file.name);
    setSourceUrl((previous) => {
      if (previous.startsWith('blob:')) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    void runPrediction(file, selectedId);
    event.target.value = '';
  }

  async function runEvidence() {
    if (!selectedFile) return;
    const token = ++analysisToken.current;
    setEvidenceBusy(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', selectedFile);
      form.append('mode', 'fast');
      form.append('occlusion', 'blur');
      const response = await fetch(apiUrl('/api/v1/analyses'), { method: 'POST', body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const job = await response.json() as { status_url: string; result_url: string };
      let resultUrl = job.result_url;
      for (let attempt = 0; attempt < 900; attempt += 1) {
        await wait(1000);
        if (analysisToken.current !== token) return;
        const statusResponse = await fetch(apiUrl(job.status_url));
        if (!statusResponse.ok) throw new Error(await responseError(statusResponse));
        const status = await statusResponse.json() as { status: string; error?: string; result_url?: string };
        resultUrl = status.result_url || resultUrl;
        if (status.status === 'failed') throw new Error(status.error || 'Evidence generation failed');
        if (status.status !== 'completed') continue;
        const resultResponse = await fetch(apiUrl(resultUrl));
        if (!resultResponse.ok) throw new Error(await responseError(resultResponse));
        const result = await resultResponse.json() as AnalysisResult;
        setEvidenceAssets({
          attribution: apiUrl(result.result.assets.attribution_overlay),
          frequency: apiUrl(result.result.assets.frequency_overlay),
        });
        setBranchValues((result.explanation.branches || []).map((branch) => ({
          label: branchLabels[branch.component] || branch.component.replaceAll('_', ' '),
          value: branch.confidence_delta,
        })));
        setEvidenceBusy(false);
        return;
      }
      throw new Error('Evidence generation timed out');
    } catch (caught) {
      if (analysisToken.current !== token) return;
      const message = caught instanceof Error ? caught.message : 'Evidence generation failed';
      setError(message === 'Failed to fetch' ? `Cannot reach the inference API at ${API_BASE}.` : message);
      setEvidenceBusy(false);
    }
  }

  const plotWidth = 1080;
  const plotHeight = 280;
  const points = useMemo(() => catalog.map((item, index) => {
    const pointConfidence = results[item.id]?.prediction.aigc_confidence;
    return {
      ...item,
      confidence: pointConfidence,
      x: 44 + index * ((plotWidth - 88) / (catalog.length - 1)),
      y: pointConfidence === undefined ? plotHeight - 42 : 20 + (1 - pointConfidence) * (plotHeight - 62),
    };
  }), [catalog, results]);
  const completedPoints = points.filter((point) => point.confidence !== undefined);
  const path = completedPoints.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
  const phaseLabel = phase === 'analyzing' ? 'INFERENCE'
    : phase === 'scanning' ? `SCAN ${scanProgress.completed}/${scanProgress.total}`
    : phase === 'error' ? 'ERROR'
    : confidence === undefined ? 'READY'
    : uncertain ? 'REVIEW' : 'DONE';

  return <main className="site-shell">
    <nav className="topbar"><a className="brand" href="#top"><span className="brand-mark">R</span><span>ROBUSTFUSION</span></a><div className="nav-links"><a href="#demo">Live demo</a><a href="#evidence">Evidence</a><a href="#transforms">Transforms</a></div><a className="nav-cta" href="#demo">Try detector ↗</a></nav>
    <div className="announcement"><span>Built for images after compression, blur, resize, noise, color edits and crops.</span><a href="#demo">Try the live detector&nbsp; ›</a></div>

    <section className="hero" id="top"><div className="hero-copy"><div className="eyebrow"><span/> TIKTOK TECHJAM 2026 · TRACK 5</div><h1>Robust evidence.<br/><em>After the edit.</em></h1><p>RobustFusion combines multiple visual cues to detect AI-generated images after compression, blur, resizing, noise, color edits and cropping.</p><div className="hero-actions"><a className="primary-button" href="#demo">Analyze an image ↗</a><a className="text-link" href="#evidence">View robustness evidence ↓</a></div><div className="proof-strip"><div><strong>16</strong><span>Stress conditions</span></div><div><strong>6</strong><span>Transform families</span></div><div><strong>Local</strong><span>Calibrated inference API</span></div></div></div><div className="hero-card"><span>MODEL · JOB 773086</span><strong>RobustFusion</strong><p>DINOv2 global and tile evidence, SigLIP semantics, and Haar-wavelet traces.</p><div><i/> Global cues</div><div><i/> Semantic cues</div><div><i/> Native-detail cues</div></div></section>

    <section className="demo section-pad" id="demo"><div className="section-heading"><div><span className="section-kicker">01 / LIVE INFERENCE</span><h2>See what survives<br/>the transformation.</h2></div><div className={`prototype-note backend-${backendState}`}><b>{backendState === 'ready' ? 'INFERENCE API READY' : backendState === 'checking' ? 'CHECKING INFERENCE API' : 'INFERENCE API OFFLINE'}</b><p>{backendState === 'ready' ? 'Choose a JPG, PNG or WEBP. The selected transform returns first; all 16 conditions then fill progressively.' : `Start the backend at ${API_BASE}, then choose an image to retry.`}</p></div></div>
      <div className={`result-shell ${phase === 'analyzing' || phase === 'scanning' ? 'is-analyzing' : ''}`}><header className="result-header" aria-live="polite"><div><span>AIGC confidence</span><strong>{confidence === undefined ? '—' : `${(confidence * 100).toFixed(2)}%`}</strong></div><p>{probabilityFake === undefined ? 'Awaiting real model output' : `P(fake) ${probabilityFake.toFixed(4)} · raw logit ${rawLogitText}`} · <b>RobustFusion</b> · {active.label}</p><div className={`state-pill ${uncertain || phase === 'error' ? 'uncertain' : ''}`}><i/>{phaseLabel}</div></header>
        <div className="image-grid"><article className="image-panel"><div className="panel-title"><span>Input after transform</span><small>{fileName}</small></div><div className="canvas-wrap"><canvas ref={sourceCanvas}/><div className="scan-line"/></div></article><article className="image-panel heat-panel"><div className="panel-title evidence-title"><div><span>{evidenceAssets ? evidenceView === 'attribution' ? 'Model attribution overlay' : 'Wavelet contribution overlay' : 'Evidence preview'}</span><small>{evidenceAssets ? 'blue: negative · magenta: positive' : 'Run evidence after detection for model-derived attribution'}</small></div><div className="evidence-tabs" role="tablist" aria-label="Evidence visualization"><button type="button" role="tab" aria-selected={evidenceView === 'attribution'} className={evidenceView === 'attribution' ? 'active' : ''} onClick={() => setEvidenceView('attribution')}>Attribution</button><button type="button" role="tab" aria-selected={evidenceView === 'frequency'} className={evidenceView === 'frequency' ? 'active' : ''} onClick={() => setEvidenceView('frequency')}>Wavelet</button></div></div><div className="canvas-wrap">{evidenceAssets ? <img className="evidence-image" src={evidenceView === 'attribution' ? evidenceAssets.attribution : evidenceAssets.frequency} alt="Model-derived contribution overlay"/> : <canvas ref={evidenceCanvas}/>}<div className="scan-line"/></div></article></div>
        <div className="upload-row"><div><strong>{active.label}</strong><span>{phase === 'scanning' ? `Robustness scan ${scanProgress.completed}/${scanProgress.total}` : active.realWorld}</span></div><div className="upload-actions"><button type="button" onClick={() => fileInput.current?.click()}>＋ Choose image</button><button type="button" className="secondary-action" disabled={!selectedFile || evidenceBusy || confidence === undefined} onClick={() => void runEvidence()}>{evidenceBusy ? 'Generating evidence…' : 'Generate evidence'}</button></div><input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={handleFile}/></div>{error && <p className="error-message" role="alert">{error}</p>}
      </div>
      <div className="transform-picker" id="transforms"><div className="picker-head"><div><span>TRANSFORMATION</span><strong>Run a competition stress condition</strong></div><small>Scores come from the calibrated job-773086 backend.</small></div><div className="transform-groups">{(['clean', 'jpeg', 'blur', 'resize', 'noise', 'color', 'crop'] as TransformKind[]).map((kind) => <div className="transform-group" key={kind}><span>{kind === 'clean' ? 'Source' : kind}</span><div>{catalog.map((item) => item.kind === kind && <button key={item.id} className={`${selectedId === item.id ? 'active' : ''} ${results[item.id] ? 'complete' : ''}`} onClick={() => selectTransform(item.id)}>{item.short}</button>)}</div></div>)}</div></div>
    </section>

    <section className="evidence section-pad" id="evidence"><div className="section-heading light"><div><span className="section-kicker">02 / EVIDENCE</span><h2>Confidence across<br/>six transform families.</h2></div><p>The foreground result appears immediately. The remaining points fill as the backend completes its deterministic robustness scan.</p></div>
      <article className="chart-card trajectory-card"><div className="chart-title"><div><span>AIGC CONFIDENCE TRAJECTORY</span><strong>Six-family stress test · {completedPoints.length}/16 complete</strong></div><b>{active.short} · {confidence === undefined ? 'pending' : confidence.toFixed(3)}</b></div><div className="line-chart-scroll"><svg className="line-chart" viewBox={`0 0 ${plotWidth} ${plotHeight}`} role="img" aria-label="AIGC confidence across sixteen transformations">{[0, 0.25, 0.5, 0.75, 1].map((value) => { const y = 20 + (1 - value) * (plotHeight - 62); return <g key={value}><line x1="44" x2={plotWidth - 44} y1={y} y2={y}/><text x="6" y={y + 4}>{value.toFixed(2)}</text></g>; })}{path && <path className="confidence-path" d={path}/>} {points.map((point) => <g key={point.id} className={`${selectedId === point.id ? 'selected' : ''} ${point.confidence === undefined ? 'pending' : ''}`} onClick={() => selectTransform(point.id)} role="button" tabIndex={0} aria-label={`${point.label}: ${point.confidence === undefined ? 'pending' : point.confidence.toFixed(3)}`}><circle cx={point.x} cy={point.y} r={selectedId === point.id ? 7 : 5}/><text className="x-label" x={point.x} y={plotHeight - 8} transform={`rotate(-38 ${point.x} ${plotHeight - 8})`}>{point.short}</text></g>)}</svg></div></article>
      <article className="chart-card branch-card"><div className="chart-title"><div><span>BRANCH COUNTERFACTUAL CONTRIBUTION</span><strong>Confidence delta after neutralization</strong></div><b>{evidenceBusy ? 'RUNNING' : 'Δ P(AIGC)'}</b></div>{branchValues.length ? <div className="branch-bars">{branchValues.map((branch) => { const width = Math.max(2, Math.min(48, Math.abs(branch.value) / 0.18 * 48)); return <div className="branch-row" key={branch.label}><span>{branch.label}</span><div className="branch-track"><i className="zero"/><b className={branch.value < 0 ? 'negative' : ''} style={branch.value < 0 ? { right: '50%', width: `${width}%` } : { left: '50%', width: `${width}%` }}/></div><strong>{branch.value >= 0 ? '+' : ''}{branch.value.toFixed(4)}</strong></div>; })}</div> : <p className="evidence-empty">Choose an image, complete detection, then select “Generate evidence” to calculate real branch counterfactuals.</p>}</article>
    </section>

    <section className="transform-table section-pad"><div className="section-heading"><div><span className="section-kicker">03 / TEST CONTRACT</span><h2>The exact image<br/>processing suite.</h2></div><p>These controls use the backend’s audited transform IDs and deterministic processing contract.</p></div><div className="table-scroll"><table><thead><tr><th>Transform</th><th>Parameters</th><th>Real-world analog</th></tr></thead><tbody><tr><td>JPEG compression</td><td>quality = 90, 70, 50, 30</td><td>Social-media re-encode, messaging</td></tr><tr><td>Gaussian blur</td><td>kernel σ = 0.5, 1.0, 2.0</td><td>Out-of-focus</td></tr><tr><td>Resize</td><td>scale 0.5× / 0.25× then upscale</td><td>Thumbnail generation</td></tr><tr><td>Gaussian noise</td><td>σ = 0.02, 0.05, 0.10</td><td>Low-light sensor noise</td></tr><tr><td>Color jitter</td><td>brightness / contrast / saturation ±20%</td><td>Filter apps, auto-enhance</td></tr><tr><td>Center crop</td><td>crop 80%</td><td>Profile-picture cropping, framing</td></tr></tbody></table></div></section>

    <section className="team-section" id="team">
      <div className="team-heading"><div><span className="section-kicker">04 / TEAM</span><h2>404 Brain Not Found</h2></div><p>Five contributors, one robust detector.</p></div>
      <div className="team-grid">{teamMembers.map((member, index) => <article className="team-member" key={member.name}><div className="portrait-frame"><img src={member.image} alt={`${member.name} portrait`}/><span>{String(index + 1).padStart(2, '0')}</span></div><h3>{member.name}</h3></article>)}</div>
    </section>
    <footer className="team-footer"><a className="closing-mark" href="#top" aria-label="Back to top">R</a></footer>
  </main>;
}
