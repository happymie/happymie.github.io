/* Display progress from the public discovery feed; original maintenance logs remain private. */
(function(){
  'use strict';
  const requests=new Map();
  const en=()=>document.documentElement.lang==='en'||document.documentElement.dataset.lang==='en';
  const text=(zh,english)=>window.AtlasLocale?.t(zh,english)||(en()?english:zh);
  const date=value=>{
    const d=new Date(value);if(!value||!Number.isFinite(d.getTime()))return text('尚无记录','No record yet');
    return new Intl.DateTimeFormat(en()?'en-GB':'zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(d)+' CST';
  };
  function count(value){return Number.isInteger(value)&&value>=0?value:0;}
  async function render(panel){
    const url='/biomed-ai-atlas/updates/feed.json';
    try{
      if(!requests.has(url)){const request=fetch(url,{cache:'no-cache',signal:AbortSignal.timeout(10000)}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();});requests.set(url,request);request.catch(()=>{if(requests.get(url)===request)requests.delete(url);});}
      const data=await requests.get(url);if(!panel.isConnected)return;
      if(data.schema!==1||!Array.isArray(data.entries)||data.entries.length>200||!Array.isArray(data.scans)||data.scans.length>60)throw Error('Invalid public discovery feed');
      const run=data.scans[0];
      if(run&&(!['completed','partial','failed'].includes(run.status)||!Number.isFinite(Date.parse(run.finishedAt))||!Number.isInteger(run.reviewed)||run.reviewed<0||!Number.isInteger(run.published)||run.published<0))throw Error('Invalid public scan');
      panel.querySelector('[data-maintenance-status]').textContent=run?text('进度已更新','Progress updated'):text('持续检索','Research in progress');
      const meta=panel.querySelector('[data-maintenance-meta]');meta.replaceChildren();
      const values=[run?text('最近检索：','Latest scan: ')+date(run.finishedAt):text('新研究按日期记录。','Research updates are recorded by date.'),text('每日北京时间 01:00 维护','Daily maintenance at 01:00 China time')];
      for(const value of values){const span=document.createElement('span');span.textContent=value;meta.appendChild(span);}
      const checked=count(run?.reviewed),published=count(run?.published);
      panel.querySelector('[data-maintenance-counts]').textContent=text(`${data.entries.length} 项研究候选 · ${checked} 项线索已核查 · ${published} 项正式新增`,`${data.entries.length} research candidates · ${checked} leads reviewed · ${published} published additions`);
      const link=panel.querySelector('[data-maintenance-link]');link.textContent=text('查看每日发现 ↗','View daily discovery ↗');link.href='/biomed-ai-atlas/updates/';
    }catch{
      if(!panel.isConnected)return;
      panel.querySelector('[data-maintenance-status]').textContent=text('每日发现','Daily discovery');
      panel.querySelector('[data-maintenance-meta]').textContent=text('打开更新记录，继续查看研究进展。','Open the update record to explore recent research.');
      panel.querySelector('[data-maintenance-link]').href='/biomed-ai-atlas/updates/';
    }
  }
  const refresh=()=>Promise.all([...document.querySelectorAll('[data-maintenance-panel]')].map(render));
  window.SiteMaintenance=Object.freeze({refresh});refresh();
  new MutationObserver(refresh).observe(document.documentElement,{attributes:true,attributeFilter:['lang','data-lang']});
})();
