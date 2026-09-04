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
