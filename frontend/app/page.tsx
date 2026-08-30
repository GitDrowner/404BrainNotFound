'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';

type TransformKind = 'clean' | 'jpeg' | 'blur' | 'resize' | 'noise' | 'color' | 'crop';
type TransformSpec = { id:string; label:string; short:string; kind:TransformKind; value:number; confidence:number; realWorld:string };

const transforms: TransformSpec[] = [
  { id:'clean', label:'Clean', short:'Clean', kind:'clean', value:0, confidence:.9822, realWorld:'Original upload' },
  { id:'jpeg-90', label:'JPEG · Q90', short:'J90', kind:'jpeg', value:.90, confidence:.9800, realWorld:'Light social re-encode' },
  { id:'jpeg-70', label:'JPEG · Q70', short:'J70', kind:'jpeg', value:.70, confidence:.9790, realWorld:'Messaging compression' },
  { id:'jpeg-50', label:'JPEG · Q50', short:'J50', kind:'jpeg', value:.50, confidence:.9500, realWorld:'Repeated reposting' },
  { id:'jpeg-30', label:'JPEG · Q30', short:'J30', kind:'jpeg', value:.30, confidence:.9200, realWorld:'Heavy platform compression' },
  { id:'blur-05', label:'Blur · σ0.5', short:'B.5', kind:'blur', value:.5, confidence:.9810, realWorld:'Mild defocus' },
  { id:'blur-10', label:'Blur · σ1.0', short:'B1', kind:'blur', value:1, confidence:.9790, realWorld:'Camera or edit blur' },
  { id:'blur-20', label:'Blur · σ2.0', short:'B2', kind:'blur', value:2, confidence:.6200, realWorld:'Strong defocus' },
  { id:'resize-05', label:'Resize · 0.5×', short:'R.5', kind:'resize', value:.5, confidence:.9870, realWorld:'Thumbnail generation' },
  { id:'resize-025', label:'Resize · 0.25×', short:'R.25', kind:'resize', value:.25, confidence:.2600, realWorld:'Aggressive thumbnailing' },
  { id:'noise-002', label:'Noise · σ0.02', short:'N.02', kind:'noise', value:.02, confidence:.8100, realWorld:'Low-light sensor noise' },
  { id:'noise-005', label:'Noise · σ0.05', short:'N.05', kind:'noise', value:.05, confidence:.8700, realWorld:'Strong sensor noise' },
  { id:'noise-010', label:'Noise · σ0.10', short:'N.10', kind:'noise', value:.10, confidence:.9300, realWorld:'Extreme sensor noise' },
  { id:'color-minus', label:'Color · −20%', short:'C−', kind:'color', value:-.2, confidence:.9740, realWorld:'Muted filter app' },
  { id:'color-plus', label:'Color · +20%', short:'C+', kind:'color', value:.2, confidence:.9580, realWorld:'Auto-enhance filter' },
  { id:'crop-80', label:'Center crop · 80%', short:'Crop', kind:'crop', value:.8, confidence:.9630, realWorld:'Profile crop or reframing' },
];

const branchBase = [
  { label:'Global spatial + wavelet', value:.0036 },
  { label:'Semantic view', value:.1717 },
  { label:'Native tiles', value:-.0109 },
];

function seededNoise(x:number, y:number) { const n = Math.sin(x*12.9898+y*78.233)*43758.5453; return n-Math.floor(n); }

function drawContain(ctx:CanvasRenderingContext2D, image:HTMLImageElement, width:number, height:number) {
  const ratio=Math.min(width/image.naturalWidth,height/image.naturalHeight), w=image.naturalWidth*ratio, h=image.naturalHeight*ratio;
  ctx.drawImage(image,(width-w)/2,(height-h)/2,w,h);
}

function applyTransform(canvas:HTMLCanvasElement, image:HTMLImageElement, spec:TransformSpec) {
  const width=960, height=620;
  canvas.width=width; canvas.height=height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true}); if(!ctx)return;
  ctx.fillStyle='#101010'; ctx.fillRect(0,0,width,height);
  if(spec.kind==='crop') {
    const sw=image.naturalWidth*spec.value, sh=image.naturalHeight*spec.value;
    ctx.drawImage(image,(image.naturalWidth-sw)/2,(image.naturalHeight-sh)/2,sw,sh,0,0,width,height);
  } else if(spec.kind==='resize') {
    const temp=document.createElement('canvas'); temp.width=Math.max(1,Math.round(width*spec.value)); temp.height=Math.max(1,Math.round(height*spec.value));
    const t=temp.getContext('2d'); if(t){t.imageSmoothingQuality='low';drawContain(t,image,temp.width,temp.height);ctx.imageSmoothingQuality='low';ctx.drawImage(temp,0,0,width,height);}
  } else {
    const blur=spec.kind==='blur'?spec.value*2:0, sat=spec.kind==='color'?1+spec.value:1, contrast=spec.kind==='color'?1+spec.value*.65:1, bright=spec.kind==='color'?1+spec.value*.35:1;
    ctx.filter=`blur(${blur}px) saturate(${sat}) contrast(${contrast}) brightness(${bright})`; drawContain(ctx,image,width,height); ctx.filter='none';
  }
  if(spec.kind==='jpeg') {
    const encoded=canvas.toDataURL('image/jpeg',spec.value), jpeg=new Image();
    jpeg.onload=()=>{ctx.clearRect(0,0,width,height);ctx.drawImage(jpeg,0,0,width,height);}; jpeg.src=encoded;
  }
  if(spec.kind==='noise') {
    const pixels=ctx.getImageData(0,0,width,height), amplitude=spec.value*255*2.15;
    for(let i=0;i<pixels.data.length;i+=4){const p=i/4,d=(seededNoise(p%width,Math.floor(p/width))-.5)*amplitude;for(let c=0;c<3;c++)pixels.data[i+c]=Math.max(0,Math.min(255,pixels.data[i+c]+d));}
    ctx.putImageData(pixels,0,0);
  }
}

export default function Home() {
  const [selected,setSelected]=useState(0), [sourceUrl,setSourceUrl]=useState('/demo-sample.svg'), [fileName,setFileName]=useState('techjam_demo_sample.svg');
  const [state,setState]=useState<'ready'|'analyzing'|'done'>('done'), [error,setError]=useState('');
  const [evidenceView,setEvidenceView]=useState<'local'|'multiscale'>('local');
  const fileInput=useRef<HTMLInputElement>(null), sourceCanvas=useRef<HTMLCanvasElement>(null), heatCanvas=useRef<HTMLCanvasElement>(null);
  const active=transforms[selected], probability=active.confidence, rawLogit=Math.log(probability/(1-probability)), uncertain=probability>.42&&probability<.68;
  const branchValues=useMemo(()=>branchBase.map((item,index)=>({...item,value:item.value*(1-selected*.006)+(index===2?selected*.0005:0)})),[selected]);

  useEffect(()=>{
    const image=new Image();
    image.onload=()=>{
      if(!sourceCanvas.current||!heatCanvas.current)return;
      applyTransform(sourceCanvas.current,image,active); applyTransform(heatCanvas.current,image,active);
      const heat=heatCanvas.current.getContext('2d'); if(!heat)return;
      if(evidenceView==='local'){
        const cols=7,rows=5,cw=heatCanvas.current.width/cols,ch=heatCanvas.current.height/rows;
        for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){const score=seededNoise(col+selected*.17,row+.5);heat.fillStyle=score>.53?`rgba(254,44,85,${.10+score*.22})`:`rgba(37,244,238,${.08+(1-score)*.22})`;heat.fillRect(col*cw,row*ch,cw,ch);heat.strokeStyle='rgba(247,247,247,.38)';heat.lineWidth=2;heat.strokeRect(col*cw,row*ch,cw,ch);}
      }else{
        const overlay=document.createElement('canvas');overlay.width=heatCanvas.current.width;overlay.height=heatCanvas.current.height;
        const o=overlay.getContext('2d');
        if(o){
          const regions=[
            {x:.12,y:.09,w:.38,h:.34,s:-.62},{x:.50,y:.08,w:.30,h:.31,s:.48},{x:.68,y:.22,w:.22,h:.24,s:.76},
            {x:.08,y:.46,w:.42,h:.34,s:-.74},{x:.40,y:.37,w:.34,h:.31,s:-.28},{x:.58,y:.54,w:.32,h:.30,s:.66},
            {x:.22,y:.68,w:.28,h:.22,s:-.45},{x:.74,y:.71,w:.22,h:.22,s:.81},
          ];
          regions.forEach((region,index)=>{
            const drift=(seededNoise(index+selected*.19,selected+.7)-.5)*.05;
            o.fillStyle=region.s>0?`rgba(254,44,85,${.18+Math.abs(region.s)*.30})`:`rgba(51,86,211,${.16+Math.abs(region.s)*.30})`;
            o.fillRect((region.x+drift)*overlay.width,(region.y-drift)*overlay.height,region.w*overlay.width,region.h*overlay.height);
          });
          heat.save();heat.filter='blur(28px)';heat.globalAlpha=.88;heat.drawImage(overlay,0,0);heat.restore();
        }
      }
      setState('done');
    };
    image.onerror=()=>{setError('The image could not be decoded. Try a JPG, PNG, WEBP or SVG file.');setState('ready');}; image.src=sourceUrl;
  },[sourceUrl,active,selected,evidenceView]);

  function selectTransform(index:number){setError('');setState('analyzing');if(index===selected){window.setTimeout(()=>setState('done'),420);}setSelected(index);}
  function handleFile(event:ChangeEvent<HTMLInputElement>){
    const file=event.target.files?.[0];if(!file)return;
    if(!file.type.startsWith('image/')){setError('Please choose an image file.');return;}
    if(file.size>12*1024*1024){setError('The prototype accepts images up to 12 MB.');return;}
    setError('');setState('analyzing');setFileName(file.name);setSourceUrl(previous=>{if(previous.startsWith('blob:'))URL.revokeObjectURL(previous);return URL.createObjectURL(file);});
  }

  const plotWidth=1080,plotHeight=280;
  const points=transforms.map((item,index)=>({x:44+index*((plotWidth-88)/(transforms.length-1)),y:20+(1-item.confidence)*(plotHeight-62),...item}));
  const path=points.map((p,i)=>`${i?'L':'M'} ${p.x} ${p.y}`).join(' ');

  return <main className="site-shell">
    <nav className="topbar"><a className="brand" href="#top"><span className="brand-mark">R</span><span>ROBUSTFUSION</span></a><div className="nav-links"><a href="#demo">Live demo</a><a href="#evidence">Evidence</a><a href="#transforms">Transforms</a></div><a className="nav-cta" href="#demo">Try detector ↗</a></nav>
    <div className="announcement"><span>Built for images after compression, blur, resize, noise, color edits and crops.</span><a href="#demo">Try the live prototype&nbsp; ›</a></div>

    <section className="hero" id="top"><div className="hero-copy"><div className="eyebrow"><span/> TIKTOK TECHJAM 2026 · TRACK 5</div><h1>Robust evidence.<br/><em>After the edit.</em></h1><p>RobustFusion combines multiple visual cues to detect AI-generated images after compression, blur, resizing, noise, color edits and cropping.</p><div className="hero-actions"><a className="primary-button" href="#demo">Analyze an image ↗</a><a className="text-link" href="#evidence">View robustness evidence ↓</a></div><div className="proof-strip"><div><strong>16</strong><span>Stress conditions</span></div><div><strong>6</strong><span>Transform families</span></div><div><strong>Local</strong><span>Browser-only prototype</span></div></div></div><div className="hero-card"><span>MODEL</span><strong>RobustFusion</strong><p>Robust AI-Generated Image Detection via Multi-Cue Fusion</p><div><i/> Global cues</div><div><i/> Semantic cues</div><div><i/> Native-detail cues</div></div></section>

    <section className="demo section-pad" id="demo"><div className="section-heading"><div><span className="section-kicker">01 / LIVE PROTOTYPE</span><h2>See what survives<br/>the transformation.</h2></div><div className="prototype-note"><b>INTERACTIVE PROTOTYPE</b><p>Images stay in this browser. Scores are fixed demonstration data until the inference API is connected.</p></div></div>
      <div className={`result-shell ${state==='analyzing'?'is-analyzing':''}`}><header className="result-header" aria-live="polite"><div><span>P(AIGC)</span><strong>{probability.toFixed(4)}</strong></div><p>raw logit {rawLogit>=0?'+':''}{rawLogit.toFixed(4)} · <b>RobustFusion</b> · {active.label}</p><div className={`state-pill ${uncertain?'uncertain':''}`}><i/>{state==='analyzing'?'ANALYZING':uncertain?'REVIEW':'DONE'}</div></header>
        <div className="image-grid"><article className="image-panel"><div className="panel-title"><span>Input after transform</span><small>{fileName}</small></div><div className="canvas-wrap"><canvas ref={sourceCanvas}/><div className="scan-line"/></div></article><article className="image-panel heat-panel"><div className="panel-title evidence-title"><div><span>{evidenceView==='local'?'Local patch contribution':'Multi-scale contribution overlay'}</span><small>{evidenceView==='local'?'cyan: lower · magenta: higher':'blue: negative · magenta: positive'}</small></div><div className="evidence-tabs" role="tablist" aria-label="Contribution visualization"><button type="button" role="tab" aria-selected={evidenceView==='local'} className={evidenceView==='local'?'active':''} onClick={()=>setEvidenceView('local')}>Local patches</button><button type="button" role="tab" aria-selected={evidenceView==='multiscale'} className={evidenceView==='multiscale'?'active':''} onClick={()=>setEvidenceView('multiscale')}>Multi-scale fusion</button></div></div><div className="canvas-wrap"><canvas ref={heatCanvas}/><div className="scan-line"/></div></article></div>
        <div className="upload-row"><div><strong>{active.label}</strong><span>{active.realWorld}</span></div><button type="button" onClick={()=>fileInput.current?.click()}>＋ Choose local image</button><input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp,image/svg+xml" hidden onChange={handleFile}/></div>{error&&<p className="error-message" role="alert">{error}</p>}
      </div>
      <div className="transform-picker" id="transforms"><div className="picker-head"><div><span>TRANSFORMATION</span><strong>Apply a competition stress condition</strong></div><small>Processing happens locally on canvas.</small></div><div className="transform-groups">{(['clean','jpeg','blur','resize','noise','color','crop'] as TransformKind[]).map(kind=><div className="transform-group" key={kind}><span>{kind==='clean'?'Source':kind}</span><div>{transforms.map((item,index)=>item.kind===kind&&<button key={item.id} className={selected===index?'active':''} onClick={()=>selectTransform(index)}>{item.short}</button>)}</div></div>)}</div></div>
    </section>

    <section className="evidence section-pad" id="evidence"><div className="section-heading light"><div><span className="section-kicker">02 / EVIDENCE</span><h2>Confidence across<br/>six transform families.</h2></div><p>Select any point or transform above to inspect the same case in the live prototype.</p></div>
      <article className="chart-card trajectory-card"><div className="chart-title"><div><span>AIGC CONFIDENCE TRAJECTORY</span><strong>Six-family stress test</strong></div><b>{active.short} · {probability.toFixed(3)}</b></div><div className="line-chart-scroll"><svg className="line-chart" viewBox={`0 0 ${plotWidth} ${plotHeight}`} role="img" aria-label="AIGC confidence across sixteen transformations">{[0,.25,.5,.75,1].map(value=>{const y=20+(1-value)*(plotHeight-62);return <g key={value}><line x1="44" x2={plotWidth-44} y1={y} y2={y}/><text x="6" y={y+4}>{value.toFixed(2)}</text></g>})}<path className="confidence-path" d={path}/>{points.map((p,index)=><g key={p.id} className={selected===index?'selected':''} onClick={()=>selectTransform(index)} role="button" tabIndex={0} aria-label={`${p.label}: ${p.confidence.toFixed(3)}`}><circle cx={p.x} cy={p.y} r={selected===index?7:5}/><text className="x-label" x={p.x} y={plotHeight-8} transform={`rotate(-38 ${p.x} ${plotHeight-8})`}>{p.short}</text></g>)}</svg></div></article>
      <article className="chart-card branch-card"><div className="chart-title"><div><span>BRANCH COUNTERFACTUAL CONTRIBUTION</span><strong>Confidence delta after neutralization</strong></div><b>Δ P(AIGC)</b></div><div className="branch-bars">{branchValues.map(branch=>{const width=Math.max(2,Math.abs(branch.value)/.18*48);return <div className="branch-row" key={branch.label}><span>{branch.label}</span><div className="branch-track"><i className="zero"/><b className={branch.value<0?'negative':''} style={branch.value<0?{right:'50%',width:`${width}%`}:{left:'50%',width:`${width}%`}}/></div><strong>{branch.value>=0?'+':''}{branch.value.toFixed(4)}</strong></div>})}</div></article>
    </section>

    <section className="transform-table section-pad"><div className="section-heading"><div><span className="section-kicker">03 / TEST CONTRACT</span><h2>The exact image<br/>processing suite.</h2></div><p>These controls mirror the competition transformation subset and its real-world analogs.</p></div><div className="table-scroll"><table><thead><tr><th>Transform</th><th>Parameters</th><th>Real-world analog</th></tr></thead><tbody><tr><td>JPEG compression</td><td>quality = 90, 70, 50, 30</td><td>Social-media re-encode, messaging</td></tr><tr><td>Gaussian blur</td><td>kernel σ = 0.5, 1.0, 2.0</td><td>Out-of-focus</td></tr><tr><td>Resize</td><td>scale 0.5× / 0.25× then upscale</td><td>Thumbnail generation</td></tr><tr><td>Gaussian noise</td><td>σ = 0.02, 0.05, 0.10</td><td>Low-light sensor noise</td></tr><tr><td>Color jitter</td><td>brightness / contrast / saturation ±20%</td><td>Filter apps, auto-enhance</td></tr><tr><td>Center crop</td><td>crop 80%</td><td>Profile-picture cropping, framing</td></tr></tbody></table></div></section>

    <section className="closing"><div className="closing-mark">R</div><p>404 BRAIN NOT FOUND · TECHJAM 2026</p><h2>RobustFusion keeps<br/><em>synthetic traces visible.</em></h2><a href="#demo">Run the prototype ↑</a></section><footer><a className="brand" href="#top"><span className="brand-mark">R</span><span>ROBUSTFUSION</span></a><span>Robust AI-Generated Image Detection via Multi-Cue Fusion</span><span>Interactive prototype · Local processing</span></footer>
  </main>;
}
