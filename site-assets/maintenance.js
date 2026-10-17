/* A small status request, independent of model payloads and training code. */
(function(){
  'use strict';
  const requests=new Map();
  const en=()=>document.documentElement.lang==='en'||document.documentElement.dataset.lang==='en';
  const text=(zh,english)=>en()?english:zh;
  const date=value=>{
    const d=new Date(value);if(!value||!Number.isFinite(d.getTime()))return text('尚无记录','No record yet');
    return new Intl.DateTimeFormat(en()?'en-GB':'zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(d)+' CST';
  };
  function count(value){return Number.isInteger(value)&&value>=0?value:0;}
  async function render(panel){
    const url=SiteSecurity.safeURL(panel.dataset.maintenanceUrl,{asset:true});
    try{
      if(!requests.has(url))requests.set(url,fetch(url,{cache:'no-cache',signal:AbortSignal.timeout(10000)}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}));
      const data=await requests.get(url);if(!panel.isConnected)return;
      if(data.schema_version!==1||!['success','partial','failed','never'].includes(data.status))throw Error('Invalid maintenance record');
      const run=data.latest_run||{};
      const labels={success:text('本次核查完成','Check complete'),partial:text('核查部分完成','Check incomplete'),failed:text('本次核查失败','Check failed'),never:text('尚未核查','No scan yet')};
      panel.querySelector('[data-maintenance-status]').textContent=labels[data.status];
      const meta=panel.querySelector('[data-maintenance-meta]');meta.replaceChildren();
      const values=[text('最近尝试：','Last attempt: ')+date(data.last_attempt_utc),text('最近成功核查：','Last successful check: ')+date(data.last_successful_scan_utc),text('最近内容变化：','Last content change: ')+date(data.last_content_change_utc),text('每天北京时间 09:00 计划核查','Scheduled daily around 09:00 China time')];
      for(const value of values){const span=document.createElement('span');span.textContent=value;meta.appendChild(span);}
      const checked=count(run.sources_checked),attempted=count(run.sources_attempted);
      const sourceCount=attempted>=checked&&attempted>0?`${checked}/${attempted}`:String(checked);
      panel.querySelector('[data-maintenance-counts]').textContent=text(`本次新增 ${count(run.added_models)} 项 · 升级 ${count(run.upgraded_models)} 项 · 成功读取 ${sourceCount} 个来源`,`This run: ${count(run.added_models)} added · ${count(run.upgraded_models)} upgraded · ${sourceCount} sources read successfully`);
      const link=panel.querySelector('[data-maintenance-link]');link.textContent=text('查看来源与更新记录 ↗','View sources and update history ↗');
      if(run.record_url){const target=SiteSecurity.safeURL(new URL(run.record_url,new URL(url)).href,{asset:true});if(target!=='#')link.href=target;}
    }catch{
      if(!panel.isConnected)return;
      panel.querySelector('[data-maintenance-status]').textContent=text('记录暂不可用','Record unavailable');
      panel.querySelector('[data-maintenance-meta]').textContent=text('可通过更新记录查看上一次已保存的检查；此状态不代表今天已完成更新。','Open the saved history to check the previous run. This does not confirm an update today.');
    }
  }
  const refresh=()=>document.querySelectorAll('[data-maintenance-panel]').forEach(render);
  window.SiteMaintenance=Object.freeze({refresh});refresh();
  new MutationObserver(refresh).observe(document.documentElement,{attributes:true,attributeFilter:['lang','data-lang']});
})();
