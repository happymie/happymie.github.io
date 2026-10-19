'use strict';
(() => {
  const $=s=>document.querySelector(s);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const safe=v=>esc(SiteSecurity.safeURL(v));
  const storage={get:(k,d)=>{try{return JSON.parse(localStorage.getItem(k))??d;}catch{return d;}},set:(k,v)=>{try{localStorage.setItem(k,JSON.stringify(v));}catch{}}};
  let lang=storage.get('biomed_lab_lang','zh'),all=[],families=[],limit=24,current=null,os='linux';
  const tx=(zh,en)=>lang==='en'?en:zh;
  const labels={upstream:['复用开源','Upstream code'],paper_port:['论文方程移植','Paper-equation port'],mechanism_demo:['机制演示','Mechanism demo'],source_route:['原版学习路径','Original-source route']};
  const label=v=>(labels[v]||labels.source_route)[lang==='en'?1:0];
  const repo='https://github.com/happymie/happymie.github.io';
  const savedStudy=storage.get('biomed_lab_studied',[]);
  const studied=new Set(Array.isArray(savedStudy)?savedStudy.filter(Number.isInteger):[]);
  const lessonCache=new Map();let lessonRequestId=0;
  const recorded=v=>['completed','historical'].includes(v?.status);
  if(!['zh','en'].includes(lang))lang='zh';
  const bytes=v=>v>=1048576?(v/1048576).toFixed(1)+' MB':Math.round(v/1024)+' KB';
  function toast(message){const n=$('#toast');n.textContent=message;n.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>n.hidden=true,2500);}
  async function copy(text){try{await navigator.clipboard.writeText(text);toast(tx('已复制','Copied'));}catch{toast(tx('请选中命令后手动复制','Select the command and copy it manually'));}}
  function translate(){
    document.documentElement.lang=lang==='en'?'en':'zh-CN';
    document.querySelectorAll('[data-zh][data-en]').forEach(n=>n.textContent=n.dataset[lang]);
    $('#language').textContent=lang==='en'?'中':'EN';
    $('#search').placeholder=tx('搜索 U-Net、配准、Transformer…','Search U-Net, registration, Transformer…');
    const f=$('#family').value;
    $('#family').innerHTML=`<option value="all">${tx('全部家族','All families')}</option>`+families.map(x=>`<option value="${esc(x.code)}">${esc(lang==='en'?x.en:x.name)}</option>`).join('');$('#family').value=f||'all';
    const s=$('#status').value;
    $('#status').innerHTML=`<option value="all">${tx('全部状态','All statuses')}</option><option value="verified">${tx('有 CPU 运行记录','CPU execution recorded')}</option>`+Object.keys(labels).map(k=>`<option value="${k}">${label(k)}</option>`).join('');$('#status').value=s;
    updateSetup();render();window.SiteMaintenance?.refresh();if(current)openLesson(current,false);
  }
  function filtered(){const q=$('#search').value.trim().toLowerCase(),f=$('#family').value,s=$('#status').value;
    return all.filter(m=>(f==='all'||m.family_code===f)&&(s==='all'||(s==='verified'?recorded(m.verification):m.implementation===s))&&(!q||[m.title,m.num,m.family_short,m.family_en,m.core_mech,m.core_mech_en,m.adapter?.mechanism].join(' ').toLowerCase().includes(q)));
  }
  function render(){
    const rows=filtered();$('#results-count').textContent=tx(`找到 ${rows.length} / ${all.length} 项课程`,`Showing ${rows.length} of ${all.length} lessons`);
    $('#model-grid').innerHTML=rows.slice(0,limit).map(m=>`<article class="model-card"><div class="card-top"><span>${String(m.num).padStart(3,'0')} / ${esc(m.year)}</span><span class="badge ${esc(m.implementation)}">${label(m.implementation)}</span></div><h3>${esc(m.title)}</h3><div class="family-name">${esc(lang==='en'?m.family_en:m.family_short)}</div><p class="card-description">${esc(lang==='en'?m.core_mech_en:m.core_mech)}</p><div class="card-bottom"><span>${recorded(m.verification)?'● CPU '+(m.verification.status==='historical'?tx('既有记录','prior record'):tx('已测','checked')):m.has_adapter?'○ '+(m.verification?.status?tx('验证未通过','Check incomplete'):tx('待验证','Awaiting check')):'○ '+tx('原版路径','Source route')} · ${bytes(m.dataset.bytes)}</span><button type="button" data-model="${m.num}" aria-label="${esc(m.title)} ${tx('打开课程','Open lesson')}">${studied.has(m.num)?'✓ ':''}${tx('打开课程 →','Open lesson →')}</button></div></article>`).join('')||`<div class="empty">${tx('没有找到课程。试试其他关键词或重置筛选。','No lessons found. Try another query or reset the filters.')}</div>`;
    $('#more').hidden=rows.length<=limit;
  }
  function updateSetup(){ $('#setup-command').textContent=tx('选择模型课程，查看配置、实现方式与运行记录。','Choose a model course to explore its configuration, implementation and recorded runs.'); $('#setup-platform').textContent='MODEL COURSES'; }
  const table=rows=>`<table class="kv"><tbody>${rows.map(([a,b])=>`<tr><td>${esc(a)}</td><td>${esc(b)}</td></tr>`).join('')}</tbody></table>`;
  function configControls(m){
    if(!m.adapter)return '';
    return `<div class="lesson-block"><h3>${tx('调整 CPU 实验','Adjust the CPU experiment')}</h3><p>${tx('缩小参数会改变模型容量。结构依赖由 Python 适配器检查。','Reducing parameters changes capacity. The Python adapter checks structural constraints.')}</p><div class="parameter-grid"><label>${tx('训练步数','Steps')}<input data-run-option="steps" value="20" type="number" min="1" max="1000"></label><label>${tx('批次大小','Batch size')}<input data-run-option="batch-size" value="2" type="number" min="2" max="8"></label><label>${tx('CPU 线程','CPU threads')}<input data-run-option="threads" value="2" type="number" min="1" max="4"></label></div><details class="advanced"><summary>${tx('模型参数与缩小配置','Model parameters and reduced config')}</summary><div class="parameter-grid">${Object.entries(m.adapter.config).filter(([k])=>k!=='batch_size').map(([k,v])=>typeof v==='number'?`<label>${esc(k)}<input data-config="${esc(k)}" type="number" step="any" value="${esc(v)}"></label>`:Array.isArray(v)?`<label>${esc(k)}<input data-config="${esc(k)}" value="${esc(JSON.stringify(v))}" aria-label="${esc(k)} JSON array"></label>`:`<label>${esc(k)}<code>${esc(JSON.stringify(v))}</code></label>`).join('')}</div></details><div class="lesson-command"><button id="copy-run">${tx('复制','Copy')}</button><pre id="run-command"></pre></div><p id="command-note" class="small"></p></div>`;
  }
  function makeCommand(){
    if(!current?.adapter)return;
    try{
      const get=(k,min,max)=>{const n=Number(document.querySelector(`[data-run-option="${k}"]`).value);if(!Number.isInteger(n)||n<min||n>max)throw Error(k);return n;};
      const steps=get('steps',1,1000),batch=get('batch-size',2,8),threads=get('threads',1,4);
      const py=os==='windows'?'.venv\\Scripts\\python.exe':'.venv/bin/python';
      let cmd=`${py} -m biomed_lab run ${String(current.num).padStart(3,'0')} --steps ${steps} --batch-size ${batch} --threads ${threads} --device cpu`;
      document.querySelectorAll('[data-config]').forEach(input=>{if(!/^[A-Za-z_]\w*$/.test(input.dataset.config))throw Error('Invalid config key');const v=JSON.parse(input.value),valid=typeof v==='number'?Number.isFinite(v):Array.isArray(v)&&v.every(x=>typeof x==='number'&&Number.isFinite(x));if(!valid)throw Error(input.dataset.config);if(JSON.stringify(v)!==JSON.stringify(current.adapter.config[input.dataset.config]))cmd+=` --set '${input.dataset.config}=${JSON.stringify(v)}'`;});
      $('#run-command').textContent=cmd;$('#copy-run').disabled=false;$('#command-note').textContent=tx('本地运行采用 120 秒、3 GiB RSS 软预算，在训练与验证阶段检查用量。测量方法见运行记录。','Local runs use a 120 s and 3 GiB RSS soft budget, checked during training and validation. Measurement details accompany each run.');
    }catch(e){$('#run-command').textContent=tx('请检查输入：','Check input: ')+e.message;$('#copy-run').disabled=true;}
  }
  async function openLesson(m,push=true){
    if(!m||!Number.isInteger(m.num))return;
    const request=++lessonRequestId;
    if(push)history.pushState(null,'',`#model=${m.num}`);
    const cached=lessonCache.get(m.num)||(m.source?m:null);
    if(cached)m=cached;
    else{
      current=m;
      $('#lesson-kicker').textContent=`MODEL ${String(m.num).padStart(3,'0')}`;
      $('#lesson-content').innerHTML=`<div class="lesson-body"><h2 id="lesson-title">${esc(m.title)}</h2><p role="status">${tx('正在载入课程详情…','Loading the lesson…')}</p></div>`;
      if(!$('#lesson-dialog').open)$('#lesson-dialog').showModal();
      try{
        const response=await fetch(`web/lessons/${String(m.num).padStart(3,'0')}.json`,{signal:AbortSignal.timeout(15000)});
        if(!response.ok)throw Error(response.status);
        const detail=await response.json();
        if(detail.num!==m.num||detail.id!==m.id||!detail.source||!Object.hasOwn(labels,detail.implementation))throw Error('Invalid lesson');
        if(lessonCache.size>=12)lessonCache.delete(lessonCache.keys().next().value);
        lessonCache.set(m.num,detail);m=detail;
      }catch{
        if(request!==lessonRequestId)return;
        $('#lesson-content').innerHTML=`<div class="lesson-body"><h2 id="lesson-title">${esc(m.title)}</h2><p>${tx('课程暂未载入，请重试。','The lesson did not load. Please retry.')}</p><button class="button" id="retry-lesson">${tx('重新载入','Try again')}</button></div>`;
        $('#retry-lesson').addEventListener('click',()=>openLesson(m,false));return;
      }
    }
    if(request!==lessonRequestId)return;
    current=m;const s=m.source,a=m.adapter,v=m.verification;
    $('#lesson-kicker').textContent=`MODEL ${String(m.num).padStart(3,'0')} / ${lang==='en'?m.family_en:m.family_short}`;
    const originalLink=`../biomed-ai-atlas/#/model/${encodeURIComponent(m.id)}`;
    const sourceNote=(!a?tx('来源课程 · 提供论文、作者实现与合成教学数据；CPU 适配待完成。','Source course · Paper, author implementation and synthetic teaching data available; CPU adapter pending.'):a.implementation==='mechanism_demo'?tx('机制实验 · 通过轻量代码观察本课程列出的核心计算。','Mechanism experiment · Explore the listed core computations with a lightweight implementation.'):tx('小规模实验 · 采用缩小配置与随机初始化权重，适合观察训练过程。','Small-scale experiment · A reduced configuration and random initial weights make the training process easier to inspect.'));
    $('#lesson-content').innerHTML=`<div class="lesson-body"><div class="lesson-badges"><span class="badge ${esc(m.implementation)}">${label(m.implementation)}</span><span class="badge">${esc(s.license_spdx||'NOASSERTION')}</span><span class="badge">${bytes(m.dataset.bytes)} SQLite</span>${recorded(v)?`<span class="badge">CPU ${v.status==='historical'?tx('既有执行记录','prior execution record'):tx('已执行验证','execution checked')}</span>`:''}</div><h2 id="lesson-title">${esc(m.title)}</h2>${a?.name&&a.name!==m.title?`<p class="small">${esc(a.name)}</p>`:''}<p class="lead">${esc(lang==='en'?m.core_mech_en:m.core_mech)}</p><div class="lesson-links"><a href="${safe(m.paper_url)}" target="_blank" rel="noopener">${tx('阅读论文 ↗','Read paper ↗')}</a><a href="${safe(s.source_url||s.repo_url||m.paper_url)}" target="_blank" rel="noopener">${tx('原始来源 ↗','Original source ↗')}</a><a href="${originalLink}">${tx('结构与数学 ↗','Structure & math ↗')}</a></div><div class="lesson-limit">${esc(sourceNote)}</div><div class="lesson-columns"><div><div class="lesson-block"><h3>${tx('本课程的计算机制','Computation in this lesson')}</h3><p>${esc(a?.mechanism||tx('沿原论文和作者实现学习本模型的完整信息条件、架构与训练方案。CPU 适配状态以本页记录为准。','Use the paper and author implementation to study the full input regime, architecture, and training procedure. This page states the CPU adapter status.'))}</p>${a?`<p></p>${a.source_url?`<p><a href="${safe(a.source_url)}" target="_blank" rel="noopener">${tx('适配器实际使用的实现 ↗','Actual adapter / library source ↗')}</a></p>`:''}`:''}</div><div class="lesson-block"><h3>${tx('实现范围与差异','Implementation scope and differences')}</h3><ul>${(a?.limitations||m.gaps||[sourceNote]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div></div><div><div class="lesson-block"><h3>${tx('数据与实验单位','Data and statistical unit')}</h3>${table([[tx('教学数据','Data'),m.dataset.kind],[tx('独立样本','Specimens'),'16 train / 4 validation / 4 test'],[tx('数据体积','Size'),bytes(m.dataset.bytes)],[tx('单位','Units'),tx('无量纲合成数据','Dimensionless synthetic data')],[tx('评价范围','Evaluation scope'),a?.evaluation_scope==='same_scene'?tx('同场景拟合诊断','Same-scene fit diagnostic'):tx('合成独立样本','Independent synthetic specimens')]])}</div><div class="lesson-block"><h3>${tx('原论文来源档案','Catalog source record')}</h3>${table([[tx('来源类型','Type'),s.source_kind],[tx('来源档案版本','Catalog revision'),s.commit?.slice(0,12)||tx('无固定版本','Unpinned')],[tx('许可证','License'),s.license_spdx||'NOASSERTION'],[tx('检查日期','Checked'),s.evidence_checked]])}${s.license_url?`<p><a href="${safe(s.license_url)}" target="_blank" rel="noopener">${tx('阅读上游许可证 ↗','Read upstream license ↗')}</a></p>`:''}</div></div></div>${configControls(m)}<div class="lesson-block"><h3>${tx('实际运行证据','Measured execution evidence')}</h3>${recorded(v)?`${v.status==='historical'?`<p class="source-note">${tx('以下为已有版本的运行记录，保留当时的配置、版本与日志。','Recorded runs from an earlier version, with their original configuration, version and logs.')}</p>`:''}${table([[tx('参数量','Parameters'),v.parameters.toLocaleString()],[tx('完成训练步数','Steps completed'),v.steps_completed],[tx('测量耗时','Measured duration'),v.elapsed_seconds.toFixed(3)+' s'],[tx('进程 RSS 峰值','Process peak RSS'),v.observed_peak_rss_mb.toFixed(1)+' MiB'],[tx('设备 / 线程 / 批次','Device / threads / batch'),`${v.device} / ${v.threads} / ${v.batch_size}`]])}<p class="small">${tx('记录来自测试环境中的短程运行。模型依次在独立进程执行；RSS 为进程峰值，包含导入、依赖与验证开销。','Short runs recorded in the test environment. Models run sequentially in separate processes; peak RSS includes imports, dependencies and validation.')}</p><table class="loss-table"><thead><tr><th>Step</th><th>Split / diagnostic</th><th>Loss</th></tr></thead><tbody>${v.losses.map(x=>`<tr><td>${x.step}</td><td>${esc(x.split)}</td><td>${x.loss.toFixed(6)}</td></tr>`).join('')}</tbody></table><p class="small"> · </p>`:`<p class="source-note">${esc(v?.error||tx('CPU 运行记录待补充。','CPU execution record pending.'))}</p>`}</div><div class="lesson-block"><h3>${tx('来源说明','Source notes')}</h3><ul>${(s.notes||[]).map(n=>`<li>${esc(n)}</li>`).join('')||`<li>${tx('固定代码版本不等于已经复现原论文训练。','The pinned version identifies the code used by this course.')}</li>`}</ul></div><div class="lesson-links"><button id="mark-studied">${studied.has(m.num)?tx('✓ 已加入学习记录','✓ In your study record'):tx('标记已学习','Mark as studied')}</button></div></div>`;
    if(a){makeCommand();$('#copy-run').addEventListener('click',()=>copy($('#run-command').textContent));document.querySelectorAll('[data-config],[data-run-option]').forEach(n=>n.addEventListener('input',makeCommand));}
    $('#mark-studied').addEventListener('click',()=>{studied.has(m.num)?studied.delete(m.num):studied.add(m.num);storage.set('biomed_lab_studied',[...studied]);render();openLesson(m,false);});
    const dialog=$('#lesson-dialog');if(!dialog.open)dialog.showModal();
  }
  function closeLesson(){lessonRequestId++;current=null;$('#lesson-dialog').close();if(location.hash.startsWith('#model='))history.replaceState(null,'','#catalog');}
  function route(){const n=Number(/^#model=(\d+)$/.exec(location.hash)?.[1]);if(n){const m=all.find(x=>x.num===n);if(m)openLesson(m,false);}else if($('#lesson-dialog').open){lessonRequestId++;current=null;$('#lesson-dialog').close();}}
  document.addEventListener('atlas:language',e=>{lang=e.detail.language==='en'?'en':'zh';storage.set('biomed_lab_lang',lang);translate();});
  let searchTimer;['search','family','status'].forEach(id=>$('#'+id).addEventListener(id==='search'?'input':'change',()=>{limit=24;clearTimeout(searchTimer);if(id==='search')searchTimer=setTimeout(render,120);else render();}));
  $('#clear').addEventListener('click',()=>{$('#search').value='';$('#family').value='all';$('#status').value='all';limit=24;render();});
  $('#more').addEventListener('click',()=>{limit+=24;render();});
  $('#model-grid').addEventListener('click',e=>{const n=e.target.closest('[data-model]');if(n)openLesson(all.find(m=>m.num===Number(n.dataset.model)));});
  document.querySelectorAll('[data-os]').forEach(n=>n.addEventListener('click',()=>{os=n.dataset.os;updateSetup();}));
  $('#copy-setup').addEventListener('click',()=>copy($('#setup-command').textContent));
  $('#close-lesson').addEventListener('click',closeLesson);$('#lesson-dialog').addEventListener('cancel',()=>{lessonRequestId++;current=null;if(location.hash.startsWith('#model='))history.replaceState(null,'','#catalog');});
  $('#lesson-dialog').addEventListener('click',e=>{if(e.target===$('#lesson-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeLesson();}});
  window.addEventListener('hashchange',route);window.addEventListener('popstate',route);
  updateSetup();
  fetch('web/catalog.json',{signal:AbortSignal.timeout(15000)}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{if(!Array.isArray(d.models)||!Array.isArray(d.families)||d.models.some(m=>!Number.isInteger(m.num)||!Object.hasOwn(labels,m.implementation)))throw Error('Invalid catalog');all=d.models;families=d.families;$('#lesson-total').textContent=all.length;$('#verified-count').textContent=all.filter(m=>recorded(m.verification)).length;$('#dataset-size').textContent=bytes(Math.max(...all.map(m=>m.dataset.bytes)));const u=all.find(m=>m.num===6);$('#hero-verification').textContent=recorded(u?.verification)?(u.verification.status==='historical'?'Prior CPU execution record':'CPU forward + backward'):'See execution record';translate();route();}).catch(()=>{$('#load-error').hidden=false;$('#results-count').textContent=tx('加载失败','Load failed');});
})();
