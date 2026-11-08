import React, { useEffect, useId, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AnimatePresence, MotionConfig, animate, motion, useMotionValue, useReducedMotion, useTransform } from 'framer-motion';
import evidence from './evidence.json';
import './style.css';

const titles = [
  'Research validation progress', 'Audit and recovery milestones',
  'Corrections across four layers', 'Independent mock calibration',
  'Native CNN training and export', 'Failures revealed by stress tests',
  'Current scientific confidence', 'The next validation milestone'
];
const notes = [
  'Snapshot: 6 November 2026. This deck summarizes the independent audit and subsequent authorized local implementation recovery. The original paper and historical results remain preserved. RadioML has not been downloaded. Local evidence does not establish paper reproduction or publication readiness.',
  'The original suite passed 205 cases despite material validation gaps. The expanded local suite passes 325 cases, with 18 upstream Keras/NumPy deprecation warnings. R006 verified hashes for 97 protected historical files and nine local assets. R007 independently recalculated counts and pairwise AUC for 25 recorded conditions. These are software and arithmetic checks, not 325 independent scientific claims.',
  'Mathematics: exact distinct-score ROC/AUC, tie handling, interpolated EER, input checks, explicit SNR and H1 distribution conventions. Physical: fixed mock receiver gain, independent H0 calibration and ring-buffer invariants. Network: positive-duration PU/SU overlap, slot completion and horizon/pending accounting. AI: actual Keras convolutional training and restorable checkpoints. Broader receiver, network and dataset claims remain unresolved.',
  'R002: 400 H0 calibration frames, 3,000 independent H0 test frames, 300 independent H1 frames. QPSK configured at +10 dB before ADC clipping. Pd=300/300; two-sided exact 95% interval [98.7779%,100%]. Pfa=49/3000; interval [1.2107%,2.1537%]. One-sided 95% Pd lower bound is 99.0064%, Pfa upper bound is 2.0678%. These results assume independent stationary trials and do not establish real SDR performance.',
  'R003b: three native Keras CNNs underwent small two-epoch synthetic smoke training with separate train, validation, H0 calibration and test seeds. All three changed convolutional weights and restored checkpoint predictions exactly on the fixture. R006 independently found finite nonzero BCE gradients in both convolution kernels in all three checkpoints, excluding weight decay as the source of those probe gradients. Nine native TFLite files execute in the interpreter. BatchNorm has 23,073 parameters rather than the paper\'s 23,201. Native ONNX runtime and edge hardware evidence remain unavailable.',
  'R005 uses a fixed synthetic GAP checkpoint and nominal calibration. Native FP32 Pfa is 5/300. Reusing the native threshold for INT8 yields 60/300 false alarms and changes 80/600 full-set decisions. A +1 dB common receiver gain applied to signal and noise preserves SNR and raises native Pfa to 170/300. Independent INT8 calibration produces only three distinct H0 score levels in this fixture and lowers 0 dB Pd from 97/100 native to 69/100 INT8. Each condition uses 300 H0 and 100 H1 frames per SNR. These findings do not falsify the differently trained original paper models.',
  'Supported locally: corrected numerical routines, specified timing/buffer properties, native graph computation and executable TFLite artifacts. Failed under the tested synthetic conditions: native-to-INT8 threshold transfer and fixed-threshold gain robustness. Inconclusive: full RadioML reproduction, network superiority/PU-receiver protection and physical device latency/energy. The data, assumptions, protocol and hardware evidence determine the scope of confidence.',
  'Recommended next milestone: obtain and verify RadioML 2018.01A, independently establish H0 noise according to a documented measurement contract, freeze train/validation/calibration/test partitions, and train one actual FP32 baseline CNN in a GPU environment. Compare it with energy detection on common sealed test trials. Report measured Pd and Pfa by SNR/modulation, exact intervals and variation across training seeds. RadioML modulation rows provide H1 and alone cannot establish Pfa. Document missing capture-group provenance. Dataset acquisition is recommended, not completed.'
];

function useNarrow() {
  const [narrow, setNarrow] = useState(window.innerWidth < 760);
  useEffect(() => { const fn = () => setNarrow(window.innerWidth < 760); window.addEventListener('resize', fn); return () => window.removeEventListener('resize', fn); }, []);
  return narrow;
}

function Counter({ value, staticMode = false }) {
  const reduced = useReducedMotion();
  const number = useMotionValue(staticMode || reduced ? value : 0);
  const display = useTransform(number, n => Math.round(n).toLocaleString('en-US'));
  useEffect(() => { if (staticMode || reduced) { number.set(value); return; } const animation = animate(number, value, { duration: .85, ease: [.16, 1, .3, 1] }); return () => animation.stop(); }, [value, number, reduced, staticMode]);
  return <motion.span>{display}</motion.span>;
}

function Flow({ nodes, id, staticMode = false, caption }) {
  const narrow = useNarrow();
  const reduced = useReducedMotion();
  const marker = `arrow-${id}-${useId().replace(/:/g, '')}`;
  const vertical = narrow && !staticMode;
  const width = vertical ? 400 : 1120;
  const height = vertical ? nodes.length * 112 + 10 : 172;
  const nodeWidth = vertical ? 340 : (1120 - (nodes.length - 1) * 32) / nodes.length;
  const h = 85;
  return <figure className="flow-figure">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={caption || nodes.map(n => n.title).join(', then ')}>
      <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="currentColor" /></marker></defs>
      {nodes.map((node, i) => {
        const x = vertical ? 30 : i * (nodeWidth + 32);
        const y = vertical ? i * 112 + 6 : 33;
        const edge = vertical ? `M200 ${y+h} V${y+112}` : `M${x+nodeWidth} ${y+h/2} H${x+nodeWidth+27}`;
        return <React.Fragment key={node.title}>
          <motion.g initial={staticMode || reduced ? false : {opacity: 0, y: 9}} animate={{opacity: 1, y: 0}} transition={{duration: .35, delay: i*.12}}>
            <rect x={x+1} y={y} width={nodeWidth-2} height={h} rx="7" className="node-bg" />
            <text x={x+nodeWidth/2} y={y+35} textAnchor="middle" className="node-title">{node.title}</text>
            <text x={x+nodeWidth/2} y={y+62} textAnchor="middle" className="node-sub">{node.sub}</text>
          </motion.g>
          {i<nodes.length-1 && <motion.path d={edge} fill="none" className="flow-edge" strokeWidth="2" markerEnd={`url(#${marker})`} initial={staticMode || reduced ? false : {pathLength: 0, opacity: 0}} animate={{pathLength:1,opacity:1}} transition={{duration:.4,delay:i*.12+.2}} />}
        </React.Fragment>;
      })}
    </svg>
    {caption && <figcaption>{caption}</figcaption>}
  </figure>;
}

function Calibration({ staticMode }) {
  const narrow = useNarrow();
  const reduced = useReducedMotion();
  const marker = `cal-${useId().replace(/:/g,'')}`;
  if (narrow && !staticMode) return <Flow id="calibration-mobile" staticMode={staticMode} nodes={[{title:'400 calibration H0 frames',sub:'Independent noise observations'},{title:'Fixed energy threshold',sub:'95% calibration confidence'},{title:'Fresh H0 and H1 tests',sub:'3,000 noise and 300 active frames'}]} />;
  const blocks = [{x:0,y:27,w:340,title:'400 H0 calibration frames',sub:'Independent noise only'}, {x:390,y:27,w:310,title:'Fixed threshold',sub:'Energy order statistic'}, {x:0,y:159,w:340,title:'Fresh H0 and H1 frames',sub:'Independent evaluation'}, {x:390,y:159,w:310,title:'Energy score',sub:'Same receiver processing'}, {x:750,y:159,w:365,title:'Sensing decision',sub:'Measured detections and false alarms'}];
  return <figure className="calibration-figure"><svg viewBox="0 0 1120 260" role="img" aria-label="Independent noise calibration produces a fixed threshold. Fresh test frames pass through the energy detector to produce sensing decisions.">
    <defs><marker id={marker} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="currentColor"/></marker></defs>
    {['M340 67 H384','M340 199 H384','M700 199 H744','M700 67 H932 V153'].map((d,i)=><motion.path key={d} d={d} fill="none" className="flow-edge" strokeWidth="2" markerEnd={`url(#${marker})`} initial={staticMode||reduced?false:{pathLength:0}} animate={{pathLength:1}} transition={{duration:.7,delay:.15*i}} />)}
    {blocks.map((b,i)=><motion.g key={b.title} initial={staticMode||reduced?false:{opacity:0}} animate={{opacity:1}} transition={{delay:i*.1,duration:.3}}><rect x={b.x+1} y={b.y} width={b.w-2} height="80" rx="7" className="node-bg"/><text x={b.x+b.w/2} y={b.y+33} textAnchor="middle" className="node-title">{b.title}</text><text x={b.x+b.w/2} y={b.y+59} textAnchor="middle" className="node-sub">{b.sub}</text></motion.g>)}
  </svg></figure>;
}

function FailureChart({ staticMode }) {
  const narrow = useNarrow() && !staticMode;
  const reduced = useReducedMotion();
  const gap = evidence.stress.models.gap;
  const nominal = gap.native.conditions.find(c=>c.gain_drift_db===0).runtime_calibrated.h0;
  const transferred = gap.int8.conditions.find(c=>c.gain_drift_db===0).native_threshold_transferred.h0;
  const shifted = gap.native.conditions.find(c=>c.gain_drift_db===1).runtime_calibrated.h0;
  const data = [{title:'FP32, nominal gain',...nominal},{title:'INT8, native threshold',...transferred},{title:'FP32, +1 dB gain',...shifted}];
  const left=narrow?18:275, plot=narrow?410:620, height=narrow?460:420;
  const x = v=>left+plot*v/60;
  return <figure className="failure-chart"><svg viewBox={`0 0 ${narrow?540:1050} ${height}`} role="img" aria-label="GAP false alarm rates are 1.67 percent at nominal gain, 20 percent after transferring the native threshold to INT8, and 56.67 percent after a one decibel gain increase.">
    <text x={left} y="22" className="axis-title">False alarms (%)</text>
    {[0,20,40,60].map(v=><g key={v}><line x1={x(v)} y1="55" x2={x(v)} y2={height-48} className="grid-line"/><text x={x(v)} y={height-18} textAnchor="middle" className="axis-label">{v}</text></g>)}
    <line x1={x(5)} x2={x(5)} y1="55" y2={height-48} className="target-line"/>
    <text x={x(5)+8} y="46" className="target-label">5% target</text>
    {data.map((d,i)=>{
      const y = narrow ? 94+i*110 : 101+i*104;
      const value=d.pfa*100;
      return <g key={d.title}>
        <text x={narrow?18:0} y={narrow?y-15:y+10} className="condition-label">{d.title}</text>
        {!narrow&&<text x="0" y={y+36} className="condition-count">{d.false_alarms}/{d.frames} H0 frames</text>}
        <motion.rect x={left} y={y} height="31" rx="2" className={i===0?'bar-reference':'bar-failure'} initial={staticMode||reduced?false:{width:0}} animate={{width:plot*value/60}} transition={{duration:.65,delay:i*.12,ease:[.16,1,.3,1]}}/>
        <text x={x(value)+12} y={y+24} className="bar-value">{value.toFixed(2)}%</text>
      </g>;
    })}
  </svg></figure>;
}

function Slide({ index, staticMode=false }) {
  const reduced = useReducedMotion();
  const enter = staticMode||reduced?{initial:false}:{initial:{opacity:0,y:14}};
  const body = (content,delay=0)=><motion.div {...enter} animate={{opacity:1,y:0}} transition={{duration:.45,delay}}>{content}</motion.div>;
  const header=(title,sub)=><header className="slide-heading"><h1>{title}</h1>{sub&&<p className="subtitle">{sub}</p>}</header>;
  if(index===0) return <div className="cover slide-body">
    <div className="cover-text"><p className="eyebrow">Cognitive radio spectrum sensing</p><h1>Research validation<br/><span>Progress update</span></h1><p className="cover-subtitle">Independent audit, implementation recovery<br/>and the evidence still needed</p><p className="date">6 November 2026</p></div>
    {body(<Flow id="cover" staticMode={staticMode} nodes={[{title:'Review the evidence',sub:'Four layers'},{title:'Correct the implementation',sub:'Reproducible local checks'},{title:'Challenge the results',sub:'Independent stress tests'}]} />,.15)}
    <p className="cover-summary">Local software confidence has improved. Full scientific reproduction remains open.</p>
  </div>;
  if(index===1) return <div className="slide-body">{header('325 software checks now pass','The audit expanded coverage beyond the original 205 passing cases')}
    {body(<div className="hero-number"><strong><Counter value={325} staticMode={staticMode}/></strong><div><h2>Passing local software cases</h2><p>Exact numerical checks, adversarial regressions<br/>and native neural execution</p></div></div>)}
    {body(<div className="milestones"><div><strong>97</strong><p>Historical evidence files<br/>preserved unchanged</p></div><div><strong>9</strong><p>Original local assets<br/>preserved unchanged</p></div><div><strong>25</strong><p>Recorded conditions<br/>independently recounted</p></div></div>,.1)}
    <p className="scope-note">Passing software tests supports specific implementation properties. Scientific claims need separate evidence.</p>
  </div>;
  if(index===2) return <div className="slide-body">{header('Corrections across four layers','The review followed the report’s mathematical, physical, network and AI/ML layers')}
    <div className="layer-list">{[
      ['01','Mathematical','Exact ROC/AUC and tied scores','Explicit gain, SNR and H1 distribution conventions'],
      ['02','Physical','Independent noise calibration','Fixed mock gain and safer streaming buffers'],
      ['03','Network','Collisions across the whole packet','Correct slot completion, pending packets and horizon'],
      ['04','AI / ML','Native CNN training and checkpoints','Executable exports with measured runtime behavior']
    ].map((r,i)=><motion.div className="layer-row" key={r[0]} {...enter} animate={{opacity:1,y:0}} transition={{duration:.35,delay:i*.08}}><span className="layer-number">{r[0]}</span><h2>{r[1]}</h2><div><h3>{r[2]}</h3><p>{r[3]}</p></div></motion.div>)}</div>
    <p className="scope-note">These corrections establish local behavior within stated assumptions.</p>
  </div>;
  if(index===3) return <div className="slide-body">{header('Independent calibration supports the mock case','Fresh calibration and evaluation frames share the same receiver processing')}
    <Calibration staticMode={staticMode}/>
    <div className="rate-results"><div><p className="metric-label">Detection probability, Pd</p><strong>100<span>%</span></strong><p>300 / 300 active frames</p><p className="interval">95% interval: 98.78%–100%</p></div><div><p className="metric-label">False alarm probability, Pfa</p><strong>1.63<span>%</span></strong><p>49 / 3,000 noise frames</p><p className="interval">95% interval: 1.21%–2.15%</p></div></div>
    <p className="scope-note">QPSK mock ADC, configured +10 dB before clipping. Exact two-sided intervals assume independent stationary trials. Hardware remains unverified.</p>
  </div>;
  if(index===4) return <div className="slide-body">{header('Native CNN training and exports now execute','Actual convolutional computation and full checkpoint restoration')}
    <Flow id="native" staticMode={staticMode} nodes={[{title:'I/Q frames',sub:'Independent synthetic partitions'},{title:'Native CNN graph',sub:'Nonzero BCE gradients'},{title:'Restored checkpoint',sub:'Zero prediction error'},{title:'TFLite runtime',sub:'Nine executable exports'}]}/>
    <table className="model-table"><thead><tr><th>Native graph</th><th>Actual parameters</th><th>Export formats executed</th></tr></thead><tbody>{[['Baseline','baseline'],['BatchNorm','batchnorm'],['GAP','gap']].map(([label,key])=><tr key={key}><td>{label}</td><td>{evidence.native.models[key].count_params.toLocaleString('en-US')}</td><td>FP32, FP16, INT8</td></tr>)}</tbody></table>
    <p className="scope-note">Two-epoch synthetic smoke runs. BatchNorm’s 23,073 parameters differ from the paper’s 23,201. Full RadioML performance remains unverified.</p>
  </div>;
  if(index===5) return <div className="slide-body">{header('Stress tests exposed sensing failures','Fixed synthetic GAP checkpoint with calibration at nominal gain')}
    <FailureChart staticMode={staticMode}/>
    <div className="failure-takeaway"><strong>97% to 69%</strong><p>Detection at 0 dB after separate INT8 calibration, <br/>compared with native FP32 on the same 100 active frames</p></div>
    <p className="scope-note">Each chart condition uses 300 H0 frames. A +1 dB common signal/noise gain preserves SNR. These results do not establish failure of the original paper.</p>
  </div>;
  if(index===6) return <div className="slide-body">{header('Current scientific confidence','The strength of each conclusion follows the scope of its evidence')}
    <div className="confidence-list">{[
      ['Supported locally','Numerical routines and native computation','Corrected metrics, specified timing properties and executable artifacts','supported'],
      ['Failed in smoke tests','INT8 threshold transfer and gain robustness','Observed counterexamples define conditions that need correction','failed'],
      ['Unverified','Full paper reproduction','Real dataset protocol, H0 provenance and repeated-seed evaluation','open'],
      ['Unverified','Network superiority and physical deployment','Receiver interference, scale, device timing and measured energy','open']
    ].map((r,i)=><motion.div className="confidence-row" key={r[1]} {...enter} animate={{opacity:1,y:0}} transition={{delay:i*.08,duration:.35}}><p className={`status-${r[3]}`}>{r[0]}</p><div><h2>{r[1]}</h2><p>{r[2]}</p></div></motion.div>)}</div>
    <p className="scope-note">The available evidence does not yet establish publication readiness.</p>
  </div>;
  return <div className="slide-body">{header('The next milestone is a controlled RadioML reproduction','One FP32 baseline CNN compared with energy detection on sealed test trials')}
    <Flow id="next" staticMode={staticMode} nodes={[{title:'Verified RadioML + H0',sub:'Document noise and sample units'},{title:'Frozen partitions',sub:'Train, validate, calibrate, test'},{title:'Native FP32 baseline',sub:'GPU training and repeated seeds'},{title:'Common held-out trials',sub:'Pd, actual Pfa and intervals'}]}/>
    <div className="next-deliverable"><p className="eyebrow">Next research deliverable</p><h2>A reproducible baseline result <br/>with a traceable data and calibration protocol</h2><p>Report performance by SNR and modulation.<br/>Use the result to guide quantization, network and hardware validation.</p></div>
    <p className="scope-note">Dataset acquisition is recommended. No RadioML dataset has been downloaded in the work summarized here. RadioML’s signal rows alone cannot establish Pfa.</p>
  </div>;
}

function App() {
  const [index,setIndex]=useState(0), [direction,setDirection]=useState(1), [notesOpen,setNotesOpen]=useState(false);
  const [scale,setScale]=useState(1);
  const stage=useRef(null), touch=useRef(null);
  const narrow=useNarrow(), reduced=useReducedMotion();
  const go=next=>{const target=Math.max(0,Math.min(7,next));setDirection(target>=index?1:-1);setIndex(target);};
  const fullscreen=()=>{if(document.fullscreenElement)document.exitFullscreen?.().catch(()=>{});else document.documentElement.requestFullscreen?.().catch(()=>{});};
  useEffect(()=>{const key=e=>{if(e.ctrlKey||e.metaKey||e.altKey)return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();go(index+1);}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(index-1);}else if(e.key==='Home'){e.preventDefault();go(0);}else if(e.key==='End'){e.preventDefault();go(7);}else if(/^[1-8]$/.test(e.key))go(Number(e.key)-1);else if(e.key.toLowerCase()==='f')fullscreen();else if(e.key.toLowerCase()==='n')setNotesOpen(v=>!v);else if(e.key==='Escape')setNotesOpen(false);};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[index]);
  useEffect(()=>{if(!stage.current)return;const observer=new ResizeObserver(([entry])=>setScale(Math.min(entry.contentRect.width/1280,entry.contentRect.height/720,1.3)));observer.observe(stage.current);return()=>observer.disconnect();},[narrow]);
  return <MotionConfig reducedMotion="user"><div className="app-shell">
    <header className="app-header"><span>RESEARCH VALIDATION</span><div><span className="snapshot">6 NOVEMBER 2026</span><button type="button" onClick={fullscreen} title="Fullscreen (F)">Fullscreen</button></div></header>
    <main ref={stage} className="stage" onTouchStart={e=>{touch.current=e.changedTouches[0].clientX;}} onTouchEnd={e=>{if(touch.current===null)return;const d=e.changedTouches[0].clientX-touch.current;if(Math.abs(d)>65)go(index+(d<0?1:-1));touch.current=null;}}>
      <div className="canvas-wrapper" style={narrow?undefined:{width:1280*scale,height:720*scale}}><div className="canvas" style={narrow?undefined:{width:1280,height:720,transform:`scale(${scale})`}}>
        <AnimatePresence initial={false} mode="wait" custom={direction}><motion.section key={index} aria-label={`Slide ${index+1} of 8: ${titles[index]}`} className="slide" custom={direction} initial={reduced?false:{opacity:0,x:direction*32}} animate={{opacity:1,x:0}} exit={reduced?undefined:{opacity:0,x:-direction*20}} transition={{duration:reduced?0:.25,ease:[.22,1,.36,1]}}><Slide index={index}/></motion.section></AnimatePresence>
      </div></div>
    </main>
    <footer className="app-footer"><div className="navigation"><button onClick={()=>go(index-1)} disabled={index===0} aria-label="Previous slide">Previous</button><span className="slide-count" aria-live="polite">{String(index+1).padStart(2,'0')} <span>/ 08</span></span><button onClick={()=>go(index+1)} disabled={index===7} aria-label="Next slide">Next</button></div><div className="slide-dots" aria-label="Jump to slide">{titles.map((title,i)=><button key={title} onClick={()=>go(i)} className={i===index?'active':''} aria-label={`Slide ${i+1}: ${title}`} aria-current={i===index?'step':undefined}><span/></button>)}</div><button onClick={()=>setNotesOpen(v=>!v)} aria-expanded={notesOpen} aria-controls="presenter-notes">Notes <span className="key-hint">N</span></button></footer>
    <div className="deck-progress" aria-hidden="true"><motion.div animate={{scaleX:(index+1)/8}} transition={{duration:reduced?0:.35}}/></div>
    <AnimatePresence>{notesOpen&&<motion.aside className="notes" id="presenter-notes" initial={reduced?false:{y:20,opacity:0}} animate={{y:0,opacity:1}} exit={{y:15,opacity:0}}><div><h2>Presenter notes</h2><button onClick={()=>setNotesOpen(false)} aria-label="Close notes">Close</button></div><h3>{titles[index]}</h3><p>{notes[index]}</p><details><summary>Evidence sources</summary>{Object.entries(evidence.sources).map(([name,source])=><p className="source" key={name}><strong>{name}</strong><br/>{source.path}<br/><span>SHA-256: {source.sha256}</span></p>)}</details></motion.aside>}</AnimatePresence>
  </div><div className="print-deck">{titles.map((title,i)=><section key={title} className="print-slide"><Slide index={i} staticMode/><p className="print-page">{i+1} / 8</p></section>)}</div></MotionConfig>;
}

createRoot(document.getElementById('root')).render(<App/>);
