const CSS = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const PAL = ['#A83246','#2E6E8E','#B07D2B','#4B6B4A','#7A4E8C','#3F5B8C'];
const NS = 'http://www.w3.org/2000/svg';
const el = (t,a={}) => { const e=document.createElementNS(NS,t);
  for(const k in a) e.setAttribute(k,a[k]); return e; };

function axes(svg,W,H,M,xs,ymax,xlab,ylab,yfmt){
  const x0=M.l, x1=W-M.r, y0=H-M.b, y1=M.t;
  const yticks=[0,.25,.5,.75,1].map(v=>v*ymax);
  yticks.forEach(v=>{
    const y=y0-(v/ymax)*(y0-y1);
    svg.appendChild(el('line',{x1:x0,x2:x1,y1:y,y2:y,stroke:'var(--grid-line)','stroke-width':1}));
    const t=el('text',{x:x0-9,y:y+4,'text-anchor':'end',class:'tick'});
    t.textContent=yfmt(v); svg.appendChild(t);
  });
  xs.forEach(k=>{
    const x=x0+((k-xs[0])/(xs[xs.length-1]-xs[0]))*(x1-x0);
    const t=el('text',{x:x,y:y0+19,'text-anchor':'middle',class:'tick'});
    t.textContent=k; svg.appendChild(t);
  });
  const xl=el('text',{x:(x0+x1)/2,y:H-4,'text-anchor':'middle',class:'axlab'});
  xl.textContent=xlab; svg.appendChild(xl);
  const yl=el('text',{x:12,y:(y0+y1)/2,'text-anchor':'middle',class:'axlab',
    transform:`rotate(-90 12 ${(y0+y1)/2})`});
  yl.textContent=ylab; svg.appendChild(yl);
  return {x0,x1,y0,y1};
}

function lineChart(mount,series,opts){
  const W=opts.w||660,H=opts.h||300,M={l:46,r:14,t:12,b:38};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  svg.setAttribute('aria-label',opts.aria||'');
  const xs=opts.xs, ymax=opts.ymax||1;
  const A=axes(svg,W,H,M,xs,ymax,opts.xlab,opts.ylab,opts.yfmt||(v=>Math.round(v*100)+'%'));
  const px=k=>A.x0+((k-xs[0])/(xs[xs.length-1]-xs[0]))*(A.x1-A.x0);
  const py=v=>A.y0-(v/ymax)*(A.y0-A.y1);
  if(opts.diag){
    const d=`M ${px(xs[0])} ${py(xs[0])} L ${px(xs[xs.length-1])} ${py(Math.min(ymax,xs[xs.length-1]))}`;
    svg.appendChild(el('path',{d,stroke:'var(--muted)','stroke-width':1.2,
      'stroke-dasharray':'4 4',fill:'none'}));
  }
  series.forEach(s=>{
    const pts=s.k.map((k,i)=>[px(k),py(s.v[i])]).filter(p=>!isNaN(p[1]));
    const d='M '+pts.map(p=>p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' L ');
    svg.appendChild(el('path',{d,stroke:s.color,'stroke-width':s.dash?1.6:2.2,fill:'none',
      'stroke-linejoin':'round','stroke-linecap':'round',
      ...(s.dash?{'stroke-dasharray':'5 4','opacity':.62}:{})}));
    if(!s.dash) pts.forEach(p=>svg.appendChild(
      el('circle',{cx:p[0],cy:p[1],r:3,fill:s.color})));
  });
  mount.appendChild(svg);
}

function legend(mount,items,note){
  mount.innerHTML = items.map(i=>
    `<span><i class="swatch" style="background:${i.color}"></i>${i.label}</span>`).join('')
    + (note?`<span class="dashnote">${note}</span>`:'');
}

function q(root, sel){ return root.querySelector(sel) || document.createElement('div'); }

function renderReport(root, D){
  if (D.width2) { renderV2(root, D); return; }   // v02 data shape
/* ---- key figures ---- */
const leakPct = (D.meta.leak_rate*100).toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
q(root,'[data-key="leak"]').textContent = leakPct+'%';
q(root,'[data-key="leak-inline"]').textContent = leakPct+'% (5 of 8,280)';

/* ---- fig: grid (6-model depth sweep) ---- */
{
  const names=Object.keys(D.grid);
  const series=names.map((n,i)=>({k:D.grid[n].immediate.k,v:D.grid[n].immediate.acc,color:PAL[i]}));
  lineChart(q(root,'[data-fig="grid"]'),series,
    {xs:[1,2,3,4,6,8],xlab:'requested serial depth  k  (hops)',ylab:'accuracy',w:660,h:300,
     aria:'Accuracy versus serial depth for six models, falling from 100 percent at one hop to under 30 percent by four hops.'});
  legend(q(root,'[data-leg="grid"]'),names.map((n,i)=>({label:n,color:PAL[i]})));
  const best=Math.max(...names.map(n=>{
    const g=D.grid[n].immediate; const i=g.k.indexOf(3); return g.acc[i];}));
  q(root,'[data-key="serial"]').textContent='1';
}

/* ---- fig: landing distributions ---- */
{
  const ks=Object.keys(D.landing), W=660, PH=96, gap=14;
  const H=ks.length*(PH+gap);
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  svg.setAttribute('aria-label','Distribution of landing positions for each requested depth; at k=2 the mode is 3 hops.');
  const maxPos=12;
  ks.forEach((k,row)=>{
    const top=row*(PH+gap), base=top+PH-22, L=54, R=W-10;
    const lab=el('text',{x:0,y:top+14,class:'tick'}); lab.textContent='k = '+k;
    svg.appendChild(lab);
    const d=D.landing[k];
    const bw=(R-L)/(maxPos+1)-3;
    for(let pos=0;pos<=maxPos;pos++){
      const idx=d.pos.indexOf(pos); const f=idx<0?0:d.frac[idx];
      const x=L+pos*((R-L)/(maxPos+1));
      const h=Math.max(f*(PH-34),0.7);
      const correct = pos===+k;
      svg.appendChild(el('rect',{x:x,y:base-h,width:bw,height:h,rx:1.5,
        fill:correct?'var(--serial)':'var(--parallel)',opacity:correct?1:.42}));
      if(f>=0.1){
        const t=el('text',{x:x+bw/2,y:base-h-4,'text-anchor':'middle',class:'tick'});
        t.textContent=Math.round(f*100); svg.appendChild(t);
      }
      if(row===ks.length-1){
        const t=el('text',{x:x+bw/2,y:base+15,'text-anchor':'middle',class:'tick'});
        t.textContent=pos; svg.appendChild(t);
      }
    }
    svg.appendChild(el('line',{x1:L,x2:R,y1:base,y2:base,stroke:'var(--rule)','stroke-width':1}));
  });
  const xl=el('text',{x:W/2,y:H-1,'text-anchor':'middle',class:'axlab'});
  xl.textContent='landing position (hops travelled)'; svg.appendChild(xl);
  q(root,'[data-fig="landing"]').appendChild(svg);
  const d2=D.landing['2'], i3=d2.pos.indexOf(3);
  q(root,'[data-key="over"]').textContent=Math.round(D.wording.default.over[3]*100)+'%';
}

/* ---- table: wording control ---- */
{
  const w=D.wording, rows=w.default.k.map((k,i)=>{
    if(k===1) return '';
    const a=w.default.over[i], b=w.explicit.over[i];
    return `<tr><td class="num">${k}</td><td class="num">${(a*100).toFixed(1)}%</td>
      <td class="num">${(b*100).toFixed(1)}%</td>
      <td class="num" style="color:var(--muted)">${((b-a)*100>=0?'+':'')}${((b-a)*100).toFixed(1)} pts</td></tr>`;
  }).join('');
  q(root,'[data-tbl="wording"]').innerHTML=rows;
}

/* ---- fig: effective hops ---- */
{
  const names=Object.keys(D.grid);
  const series=names.map((n,i)=>({k:D.grid[n].immediate.k,v:D.grid[n].immediate.eff,color:PAL[i]}));
  lineChart(q(root,'[data-fig="eff"]'),series,
    {xs:[1,2,3,4,6,8],ymax:9,xlab:'requested serial depth  k  (hops)',ylab:'mean landing position',
     w:660,h:300,diag:true,yfmt:v=>v.toFixed(0),
     aria:'Mean landing position versus requested depth, above the diagonal at low k and below at high k.'});
  legend(q(root,'[data-leg="eff"]'),names.map((n,i)=>({label:n,color:PAL[i]})),
    'dashed = perfectly controlled walk');
}

/* ---- fig: task comparison panels ---- */
{
  const C={serial:'var(--serial)',parallel:'var(--parallel)',parallel_count:'var(--count)'};
  const mount=q(root,'[data-fig="task"]');
  Object.keys(D.taskcmp).forEach(model=>{
    const div=document.createElement('div'); div.className='panel';
    div.innerHTML=`<div class="ptitle">${model}</div>`;
    const host=document.createElement('div');
    const series=[];
    ['serial','parallel','parallel_count'].forEach(t=>{
      const d=D.taskcmp[model][t];
      series.push({k:d.immediate.k,v:d.immediate.acc,color:C[t]});
      series.push({k:d.filler.k,v:d.filler.acc,color:C[t],dash:true});
    });
    lineChart(host,series,{xs:[2,3,4,6,8],xlab:'k (lookups)',ylab:'accuracy',w:330,h:230,
      aria:`${model}: parallel tasks stay high while the serial task collapses.`});
    div.appendChild(host); mount.appendChild(div);
  });
  legend(q(root,'[data-leg="task"]'),[
    {label:'serial (depth k)',color:'var(--serial)'},
    {label:'parallel (depth 2)',color:'var(--parallel)'},
    {label:'parallel-count (depth 2, no shortcut)',color:'var(--count)'}],
    'dashed = 100 filler tokens');
}

/* ---- fig: width sweep to k=24 ---- */
{
  const C={serial:'var(--serial)',parallel:'var(--parallel)',parallel_count:'var(--count)'};
  const mount=q(root,'[data-fig="width"]');
  const xs=[2,4,8,12,16,20,24];
  Object.keys(D.width).forEach(model=>{
    const div=document.createElement('div'); div.className='panel';
    div.innerHTML=`<div class="ptitle">${model}</div>`;
    const host=document.createElement('div');
    const series=['serial','parallel','parallel_count'].map(t=>
      ({k:D.width[model][t].k,v:D.width[model][t].acc,color:C[t]}));
    series.push({k:xs,v:xs.map(k=>1/k),color:'var(--muted)',dash:true});
    lineChart(host,series,{xs:xs,xlab:'k (lookups)',ylab:'accuracy',w:330,h:230,
      aria:`${model}: parallel accuracy stays far above chance out to 24 lookups while serial sits at the floor.`});
    div.appendChild(host); mount.appendChild(div);
  });
  legend(q(root,'[data-leg="width"]'),[
    {label:'serial (depth k)',color:'var(--serial)'},
    {label:'parallel (depth 2)',color:'var(--parallel)'},
    {label:'parallel-count (depth 2)',color:'var(--count)'}],
    'dashed = chance for the parallel task (1/k)');

  const P=Object.keys(D.width).map(m=>D.width[m].parallel.acc.slice(-1)[0]);
  const S=Object.keys(D.width).map(m=>D.width[m].serial.acc.slice(-1)[0]);
  q(root,'[data-cap="width"]').textContent =
    `Width does not break inside the tested range. At 24 independent lookups the parallel task still scores `
    + `${Math.round(Math.min(...P)*100)}–${Math.round(Math.max(...P)*100)}% against a chance rate of 4%, `
    + `while the serial task sits at ${Math.round(Math.max(...S)*100)}% or below from k=4 onward. `
    + `The count variant declines steadily, placing the cost of aggregating many results between the two.`;
  q(root,'[data-key="parallel"]').textContent='24+';
}

}


function renderV2(root, D){
  if (D.facts) { renderV3(root, D); return; }
  if (D.transcripts) buildBrowser(root, D.transcripts);
  const C = {parallel:'var(--parallel)', parallel_count:'var(--count)'};
  // --- width sweep to k=64, one panel per model
  const wm = q(root,'[data-fig="width2"]');
  Object.keys(D.width2).forEach(model => {
    const div = document.createElement('div'); div.className = 'panel';
    div.innerHTML = `<div class="ptitle">${model}</div>`;
    const host = document.createElement('div');
    const xs = D.width2[model].parallel.k;
    const series = ['parallel','parallel_count']
      .filter(t => D.width2[model][t])
      .map(t => ({k: D.width2[model][t].k, v: D.width2[model][t].acc, color: C[t]}));
    series.push({k: xs, v: xs.map(k => 1/k), color:'var(--muted)', dash:true});
    lineChart(host, series, {xs, xlab:'k (independent lookups)', ylab:'accuracy',
      w:330, h:230, aria:`${model}: parallel accuracy stays far above chance to k=64.`});
    div.appendChild(host); wm.appendChild(div);
  });
  legend(q(root,'[data-leg="width2"]'), [
    {label:'parallel (depth 2)', color:'var(--parallel)'},
    {label:'parallel-count (no shortcut)', color:'var(--count)'}], 'dashed = chance (1/k)');

  // --- bounds table
  const tb = q(root,'[data-tbl="bounds"]');
  tb.innerHTML = Object.entries(D.bounds).map(([m,b]) =>
    `<tr><td>${m}</td><td>${b.arch}</td><td class="num">${b.layers}</td>
     <td class="num">${b.T4096.toLocaleString()}</td>
     <td class="num">${b.T32768.toLocaleString()}${b.published ? ` <span style="color:var(--muted)">(pub. ${b.published.toLocaleString()})</span>`:''}</td></tr>`).join('');

  // --- filler: grouped bars, one group per k
  const fm = q(root,'[data-fig="filler"]');
  const ks = Object.keys(D.filler.acc).sort((a,b)=>a-b);
  const W=660,H=300,M={l:46,r:14,t:12,b:52};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  svg.setAttribute('aria-label','Filler token budget has no effect on accuracy at any depth.');
  const x0=M.l,x1=W-M.r,y0=H-M.b,y1=M.t;
  [0,.25,.5,.75,1].forEach(v=>{
    const y=y0-v*(y0-y1);
    svg.appendChild(el('line',{x1:x0,x2:x1,y1:y,y2:y,stroke:'var(--grid-line)','stroke-width':1}));
    const t=el('text',{x:x0-9,y:y+4,'text-anchor':'end',class:'tick'});
    t.textContent=Math.round(v*100)+'%'; svg.appendChild(t);
  });
  const gw=(x1-x0)/ks.length;
  ks.forEach((k,gi)=>{
    const budgets=Object.keys(D.filler.acc[k]).sort((a,b)=>a-b);
    const bw=Math.min(26,(gw-14)/budgets.length);
    budgets.forEach((b,bi)=>{
      const acc=D.filler.acc[k][b];
      const x=x0+gi*gw+8+bi*bw;
      const h=Math.max(acc*(y0-y1),1);
      svg.appendChild(el('rect',{x,y:y0-h,width:bw-3,height:h,rx:2,
        fill: b==='0' ? 'var(--serial)' : 'var(--parallel)',
        opacity: b==='0' ? 1 : 0.35+0.2*bi}));
      const lab=el('text',{x:x+(bw-3)/2,y:y0+12,'text-anchor':'middle',class:'tick'});
      lab.textContent = b==='0' ? 'none' : b; svg.appendChild(lab);
    });
    const kl=el('text',{x:x0+gi*gw+gw/2,y:y0+30,'text-anchor':'middle',class:'axlab'});
    kl.textContent=`k=${k}`; svg.appendChild(kl);
  });
  const yl=el('text',{x:12,y:(y0+y1)/2,'text-anchor':'middle',class:'axlab',
    transform:`rotate(-90 12 ${(y0+y1)/2})`});
  yl.textContent='accuracy'; svg.appendChild(yl);
  const xl=el('text',{x:W/2,y:H-4,'text-anchor':'middle',class:'axlab'});
  xl.textContent='filler tokens appended before the forced answer'; svg.appendChild(xl);
  fm.appendChild(svg);
}


function buildBrowser(root, TX){
  const ctl = root.querySelector('[data-tx="controls"]');
  const list = root.querySelector('[data-tx="list"]');
  if (!ctl || !list) return;
  const uniq = f => [...new Set(TX.map(f))].sort();
  ctl.innerHTML =
    `<input type="search" placeholder="search prompts and responses…" aria-label="Search transcripts">
     <select data-f="task"><option value="">all tasks</option>${uniq(t=>t.task).map(v=>`<option>${v}</option>`).join('')}</select>
     <select data-f="cond"><option value="">all conditions</option>${uniq(t=>t.cond).map(v=>`<option>${v}</option>`).join('')}</select>
     <select data-f="k"><option value="">all depths</option>${uniq(t=>t.k).map(v=>`<option>${v}</option>`).join('')}</select>
     <select data-f="ok"><option value="">correct or not</option><option value="1">correct only</option><option value="0">wrong only</option></select>
     <span class="tx-count"></span>`;

  const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const hl = (s, q) => {
    const e = esc(s);
    if (!q) return e;
    return e.replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'gi'), m=>`<mark>${m}</mark>`);
  };

  function draw(){
    const q = ctl.querySelector('input').value.trim();
    const f = {};
    ctl.querySelectorAll('select').forEach(s => { if (s.value) f[s.dataset.f] = s.value; });
    const hits = TX.filter(t =>
      (!f.task || t.task === f.task) && (!f.cond || t.cond === f.cond) &&
      (!f.k || String(t.k) === f.k) && (!f.ok || String(t.ok) === f.ok) &&
      (!q || (t.prompt + ' ' + t.completion + ' ' + t.gold + ' ' + t.pred).toLowerCase().includes(q.toLowerCase())));
    ctl.querySelector('.tx-count').textContent = `${hits.length} of ${TX.length}`;
    if (!hits.length){ list.innerHTML = '<div class="tx-empty">No transcripts match.</div>'; return; }
    list.innerHTML = hits.slice(0, 60).map(t => `
      <details class="tx-item">
        <summary class="tx-head">
          <span class="tx-pill ${t.ok ? 'tx-ok' : 'tx-no'}">${t.ok ? 'correct' : 'wrong'}</span>
          <span>${t.task}</span><span>${t.cond}</span><span>k=${t.k}</span>
          <span>expected ${esc(t.gold)}</span><span>answered ${esc(t.pred || '—')}</span>
          ${t.eff !== null && t.eff !== undefined && t.task === 'serial' ? `<span>landed ${t.eff} hops</span>` : ''}
          ${t.toks > 0 ? `<span>${t.toks} tokens out</span>` : ''}
        </summary>
        <div class="tx-body">
          <p class="tx-label">Prompt as the model received it</p>
          <pre>${hl(t.prompt, q)}</pre>
          <p class="tx-label">Model response</p>
          <pre>${hl(t.completion, q) || '<em>(empty)</em>'}</pre>
        </div>
      </details>`).join('') +
      (hits.length > 60 ? `<div class="tx-empty">Showing the first 60 of ${hits.length}. Narrow the search to see more.</div>` : '');
  }
  ctl.addEventListener('input', draw);
  draw();
}


function renderV3(root, D){
  const models = Object.keys(D.facts);
  // depth curves, one panel per model
  const mount = q(root,'[data-fig="facts"]');
  models.forEach(m => {
    const rows = D.facts[m];
    const div = document.createElement('div'); div.className='panel';
    div.innerHTML = `<div class="ptitle">${m}</div>`;
    const host = document.createElement('div');
    const xs = rows.map(r=>r.k);
    lineChart(host, [
      {k:xs, v:rows.map(r=>r.cot), color:'var(--count)'},
      {k:xs, v:rows.map(r=>r.immediate), color:'var(--serial)'},
      {k:xs, v:rows.map(r=>r.filler), color:'var(--parallel)', dash:true},
    ], {xs, xlab:'hops of real-world fact composition', ylab:'accuracy', w:330, h:230,
        aria:`${m}: no-CoT accuracy holds to three hops then collapses.`});
    div.appendChild(host); mount.appendChild(div);
  });
  legend(q(root,'[data-leg="facts"]'), [
    {label:'chain of thought', color:'var(--count)'},
    {label:'no CoT', color:'var(--serial)'},
    {label:'no CoT + filler', color:'var(--parallel)'}]);

  // filler table
  const tb = q(root,'[data-tbl="filler"]');
  const rows = D.facts[models[0]];
  tb.innerHTML = rows.map(r => {
    const sig = r.p < 0.001;
    const other = D.facts[models[1]] ? D.facts[models[1]].find(x=>x.k===r.k) : null;
    return `<tr><td class="num">${r.k}</td><td class="num">${r.immediate.toFixed(3)}</td>
      <td class="num">${r.filler.toFixed(3)}</td>
      <td class="num" style="color:${sig?'var(--parallel)':'var(--muted)'}">
        ${r.diff>=0?'+':''}${r.diff.toFixed(3)}${other?` / ${other.diff>=0?'+':''}${other.diff.toFixed(3)}`:''}</td>
      <td class="num">${r.p<0.001?'&lt;0.001':r.p.toFixed(3)}</td></tr>`;
  }).join('');

  // lens: best rank per hop, log-ish bars
  const lm = q(root,'[data-fig="lens"]');
  const br = D.lens.best_rank.filter(r=>r.solved===1);
  const ks = [...new Set(br.map(r=>r.k))].sort();
  const W=660,H=250,M={l:120,r:20,t:14,b:44};
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,role:'img'});
  svg.setAttribute('aria-label','The un-emitted bridge entity reaches rank 0 while the control reaches 57 to 88.');
  const items=[];
  ks.forEach(k => br.filter(r=>r.k===k).sort((a,b)=>a.target.localeCompare(b.target))
     .forEach(r => items.push({label:`k=${k} ${r.target}`, v:r.rank, t:r.target})));
  const max=Math.max(...items.map(i=>i.v),100);
  const bh=(H-M.t-M.b)/items.length;
  items.forEach((it,i)=>{
    const y=M.t+i*bh;
    const w=Math.max((Math.log10(it.v+1)/Math.log10(max+1))*(W-M.l-M.r),2);
    svg.appendChild(el('rect',{x:M.l,y:y+2,width:w,height:bh-5,rx:2,
      fill: it.t==='null' ? 'var(--serial)' : 'var(--parallel)'}));
    const lb=el('text',{x:M.l-8,y:y+bh/2+3,'text-anchor':'end',class:'tick'});
    lb.textContent=it.label; svg.appendChild(lb);
    const vl=el('text',{x:M.l+w+6,y:y+bh/2+3,class:'tick'});
    vl.textContent='rank '+Math.round(it.v); svg.appendChild(vl);
  });
  const xl=el('text',{x:W/2,y:H-6,'text-anchor':'middle',class:'axlab'});
  xl.textContent='median best rank in a 248,000-token vocabulary (log scale, shorter is stronger)';
  svg.appendChild(xl); lm.appendChild(svg);

  // layer profile: where in the stack the un-emitted bridge entity emerges
  const lp = q(root,'[data-fig="layerprof"]');
  if (lp && D.lens.by_layer) {
    const byK = {};
    D.lens.by_layer.forEach(r => { (byK[r.k] = byK[r.k] || []).push(r); });
    Object.keys(byK).sort().forEach(k => {
      const rows = byK[k];
      const div = document.createElement('div'); div.className='panel';
      div.innerHTML = `<div class="ptitle">${k} hops</div>`;
      const host = document.createElement('div');
      const layers = [...new Set(rows.map(r=>r.layer))].sort((a,b)=>a-b);
      const series = [
        {t:'hop1', color:'var(--parallel)'},
        {t:'hop2', color:'var(--count)'},
        {t:'null', color:'var(--serial)'},
      ].filter(s => rows.some(r=>r.target===s.t)).map(s => ({
        k: layers,
        v: layers.map(L => {
          const m = rows.find(r=>r.layer===L && r.target===s.t);
          return m ? 1 - Math.log10(m.rank+1)/Math.log10(250000) : NaN;
        }),
        color: s.color}));
      lineChart(host, series, {xs:layers, xlab:'layer', ylab:'readout strength',
        w:330, h:230, yfmt:v=>'', aria:`${k} hops: the bridge entity strengthens through the layers.`});
      div.appendChild(host); lp.appendChild(div);
    });
    legend(q(root,'[data-leg="layerprof"]'), [
      {label:'hop1 — un-emitted bridge', color:'var(--parallel)'},
      {label:'hop2 — the answer', color:'var(--count)'},
      {label:'null — unrelated control', color:'var(--serial)'}],
      'higher = better vocabulary rank (log)');
  }
  if (D.transcripts) buildBrowser(root, D.transcripts);
  const kk=q(root,'[data-key="k"]'); if(kk) kk.textContent='3';
}
