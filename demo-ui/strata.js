'use strict';
const names = {default:'Strata default',cpu_4:'4 CPU workers',cpu_7:'7 CPU workers',adaptive_160:'Adaptive cache · 160',no_pcie:'No PCIe offload'};
const colors = {default:'var(--teal)',cpu_4:'var(--amber)',cpu_7:'#aa7daa',adaptive_160:'#7e8dcc',no_pcie:'var(--muted)'};
let snapshot=null, length=32768, stage="tuning", selected=null;
const $=id=>document.getElementById(id);
const format=(n,d=1)=>Number.isFinite(n)?n.toLocaleString('en-US',{maximumFractionDigits:d}):'—';
const median=xs=>{const a=[...xs].sort((a,b)=>a-b),m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2;};
function element(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function rows(){return (snapshot?.experiments||[]).filter(e=>e.result);}
function renderChart(){
  const chart=$('chart'), metric=$('metric').value, factor=metric==='expert_hit_rate'?100:1;
  chart.replaceChildren();
  const groups=Object.keys(names).map(policy=>({policy,data:rows().filter(e=>e.result.policy===policy&&e.result.prompt_tokens===length&&e.result.stage===stage&&Number.isFinite(e.result[metric]))}));
  const values=groups.flatMap(g=>g.data.map(e=>e.result[metric]*factor));
  const max=Math.max(...values,1)*1.08;
  $('chart-scope').textContent=`${length.toLocaleString()} actual input tokens · ${stage==='tuning'?'tuning grid':'fresh confirmation prompts'} · ${metric==='first_token_seconds'?'lower is better':'higher is better'}`;
  if(!values.length){chart.append(element('p','No completed measurements for this context and metric yet.','empty'));return;}
  for(const {policy,data} of groups){
    if(!data.length)continue;
    const vals=data.map(e=>e.result[metric]*factor), row=element('div',undefined,'chart-row');
    const label=element('div',names[policy],'chart-label');label.append(element('small',`${data.length} measured · ${data.filter(e=>e.result.correct).length} passing`));
    const track=element('div',undefined,'track');track.style.setProperty('--color',colors[policy]);
    const bar=element('div',undefined,'bar');bar.style.setProperty('--scale',String(median(vals)/max));track.append(bar);
    data.forEach(e=>{const dot=element('span',undefined,'dot'+(e.result.correct?'':' fail'));dot.style.setProperty('--x',`${e.result[metric]*factor/max*100}%`);dot.title=`Block ${e.result.block+1}: ${format(e.result[metric]*factor)} · ${e.result.correct?'passed':'failed'}`;track.append(dot);});
    const value=element('div',format(median(vals))+(metric==='expert_hit_rate'?'%':''),'chart-value');
    value.append(element('small',`${format(Math.min(...vals))}–${format(Math.max(...vals))}`));row.append(label,track,value);chart.append(row);
  }
}
function openRecord(id){
  selected=id;const experiment=rows().find(e=>e._id===id);if(!experiment)return;
  const r=experiment.result, root=$('record');root.hidden=false;root.replaceChildren();
  root.append(element('h3',`${names[r.policy]} · ${r.stage} block ${r.block+1}`));
  root.append(element('p',`${experiment._id} · attempt ${experiment.attempt}`,'small'));
  const grid=element('div',undefined,'record-grid');
  for(const [title,body] of [['Raw model answer',r.raw_output||JSON.stringify(r.error,null,2)],['Engine counters and quality checks',JSON.stringify({telemetry:r.telemetry,quality:r.quality,input_verified:r.input_verified},null,2)]]){const part=element('div');part.append(element('h3',title),element('pre',body));grid.append(part);}
  root.append(grid,element('h3','Exact command'),element('pre',JSON.stringify(r.command)));
  const decision=(snapshot.decisions||[]).find(d=>d.candidate===experiment._id.split(':').slice(1).join(':'));
  root.append(element('p',`Decision consulted ${decision?.retrieved_experiment_ids?.length||0} earlier experiment records. ${decision?.hypothesis||''}`,'muted'));
  if(r.prompt?.artifact){const a=element('a','Download tokenized input and oracle (.json.gz)');a.href='../'+r.prompt.artifact;root.append(a);}
}
function render(){
  const run=snapshot.run, experiments=rows(), results=experiments.map(e=>e.result), audit=snapshot.atlas_audit||{};
  $('status').textContent=`${run.status==='completed'?'Completed':'Running'} · Atlas run ${run._id} · ${run.context?.reasoning?'reasoning enabled':'reasoning disabled'} · snapshot ${audit.read_back_at||'unknown'}`;
  $('trials').textContent=experiments.length;
  $('trials-detail').textContent=`${snapshot.decisions?.length||0} decisions with evidence links`;
  const measured=results.filter(r=>r.engine_success&&r.input_verified).map(r=>r.telemetry?.prompt_tokens||0);
  $('context').textContent=measured.length?format(Math.max(...measured),0):'—';
  $('quality').textContent=`${results.filter(r=>r.quality?.passed).length} / ${results.length}`;
  const tuning=results.filter(r=>r.stage==='tuning'&&Number.isFinite(r.decode_tps));
  const medians=Object.keys(names).map(policy=>{const arm=tuning.filter(r=>r.policy===policy);return {policy,n:arm.length,rate:arm.length?median(arm.map(r=>r.decode_tps)):null};}).filter(a=>a.n===run.repeats);
  const baseline=medians.find(a=>a.policy==='default'), fastest=[...medians].sort((a,b)=>b.rate-a.rate)[0];
  if(baseline&&fastest){$('gain').textContent=`+${format(100*(fastest.rate/baseline.rate-1))}%`;$('gain-detail').textContent=`${names[fastest.policy]} · descriptive speed, not a validated recommendation`;}
  $('quality-detail').textContent=`${results.filter(r=>r.quality?.matched_settings===8).length}/${results.length} retrieved all 8 settings; ${results.filter(r=>r.quality?.total_correct).length}/${results.length} passed the sum`;
  const verdict=$('verdict');verdict.replaceChildren();
  verdict.append(element('strong',run.status==='completed'?`Recommendation: ${names[run.recommendation]||'No validated policy'}`:'Experiments in progress'));
  verdict.append(document.createTextNode(run.status==='completed'?(run.conclusion+'. Every failed candidate remains in the ledger.'):run.selection?`Tuning selected ${names[run.selection.policy]}. Fresh 32K prompts decide whether it is retained.`:'Five policy settings are being measured on the same 32K inputs. Selection waits for all repeats.'));
  $('audit').textContent=`${audit.experiments??0} records · ${run.reused_experiment_ids?.length||0} recovered without rerun · ${audit.dangling_references??'—'} dangling references`;
  const body=$('ledger');body.replaceChildren();
  experiments.forEach((e,i)=>{const r=e.result,tr=element('tr'),cell=element('td'),button=element('button',`${r.stage==='tuning'?'Tune':'Confirm'} ${i+1}`);button.type='button';button.addEventListener('click',()=>openRecord(e._id));cell.append(button);tr.append(cell);
    for(const text of [names[r.policy],format(r.prompt_tokens,0),format(r.telemetry?.generated,0),format(r.decode_tps),format(r.resources?.gpu_peak_mib,0)])tr.append(element('td',text));
    tr.append(element('td',r.correct?'Passed':r.engine_success?'Quality failed':'Execution failed',r.correct?'pass':'fail'));body.append(tr);
  });
  renderChart();if(selected)openRecord(selected);
}
async function refresh(){
  try{const response=await fetch('../evidence/strata/latest.json',{cache:'no-store'});if(!response.ok)throw Error('No exported trials yet');snapshot=await response.json();render();}
  catch(error){$('status').textContent=snapshot?'Refresh unavailable; showing the last loaded snapshot.':'Strata installation or first trial in progress. No measurements have been published yet.';}
}
document.querySelectorAll('[data-stage]').forEach(button=>button.addEventListener('click',()=>{stage=button.dataset.stage;document.querySelectorAll('[data-stage]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderChart();}));
$('metric').addEventListener('change',renderChart);
refresh();setInterval(()=>{if(!document.hidden)refresh();},15000);
