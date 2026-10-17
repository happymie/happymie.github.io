/* The reference and atlas share the exact same model body; load it on demand. */
(function(){
  'use strict';
  const pending=new Map(),queue=[];let active=0,observer;
  const status=document.getElementById('reference-load-status');
  const allButton=document.getElementById('reference-load-all');
  const isId=id=>/^architecture_model_\d{3}_[a-z0-9_]+$/.test(id);
  const loaded=new Set();
  function pump(){
    while(active<3&&queue.length){const work=queue.shift();active++;work().finally(()=>{active--;pump();});}
  }
  function setGalleryImage(card,source,alt){
    const thumb=card?.querySelector('.gallery-thumb-img');
    const url=SiteSecurity.safeURL(source,{asset:true});
    if(!thumb||url==='#')return;
    thumb.loading='lazy';thumb.decoding='async';
    thumb.src=url;if(alt)thumb.alt=alt;
  }
  // Cover paths are verified metadata. Native lazy images do not fetch every model body.
  for(const model of window.MODELS_META||[]){
    if(isId(model.id))setGalleryImage(document.getElementById(model.id),model.image);
  }
  function enhance(body){
    for(const el of body.querySelectorAll('.circuit-node,.arch-figure-box')){el.tabIndex=0;el.setAttribute('role','button');}
    for(const img of body.querySelectorAll('img')){img.loading='lazy';img.decoding='async';}
    const figure=body.querySelector('.arch-figure-img');
    if(figure)setGalleryImage(body.closest('.model-card'),figure.getAttribute('src'),figure.getAttribute('alt'));
  }
  function load(id){
    if(!isId(id))return Promise.reject(Error('Invalid model ID'));
    const container=document.querySelector('[data-lazy-model="'+id+'"]');
    if(!container||loaded.has(id))return Promise.resolve(true);
    if(pending.has(id))return pending.get(id);
    const job=new Promise((resolve,reject)=>{
      queue.push(async()=>{
        container.setAttribute('aria-busy','true');
        try{
          const response=await fetch('models/'+encodeURIComponent(id)+'.json',{signal:AbortSignal.timeout(15000)});
          if(!response.ok)throw Error('HTTP '+response.status);
          const data=await response.json(),fragment=SiteSecurity.fragment(data.html);
          const body=fragment.querySelector('.model-full-body');
          if(!body)throw Error('Model body missing');
          container.replaceChildren(...body.childNodes);
          container.removeAttribute('data-lazy-model');container.dataset.loadedModel=id;
          container.removeAttribute('aria-live');enhance(container);loaded.add(id);observer?.unobserve(container);
          if(typeof window.selectArchNode==='function')window.selectArchNode(id,0);
          resolve(true);
        }catch(error){
          const p=container.querySelector('p');if(p)p.textContent='正文暂未载入，请重试。 / The model text could not load. Please retry.';
          reject(error);
        }finally{container.removeAttribute('aria-busy');pending.delete(id);}
      });pump();
    });pending.set(id,job);return job;
  }
  const quiet=id=>load(id).catch(()=>false);
  // Wrappers preserve each original interaction after its body becomes available.
  for(const name of ['openLightbox','expandSingleCardModal']){
    const original=window[name];window[name]=async function(id){if(await quiet(id))return original.call(this,id);};
  }
  const jump=window.jumpToModelCard;
  window.jumpToModelCard=function(id){quiet(id);return jump.call(this,id);};
  const compare=window.renderCompareModalBody;let compareGeneration=0;
  window.renderCompareModalBody=async function(){
    const generation=++compareGeneration;
    await Promise.all((APP_STATE.compareList||[]).map(quiet));
    if(generation===compareGeneration)return compare.call(this);
  };
  document.addEventListener('click',event=>{
    const button=event.target.closest('[data-load-model]');if(button)quiet(button.dataset.loadModel);
  });
  allButton.addEventListener('click',async()=>{
    allButton.disabled=true;allButton.textContent='正在载入… / Loading…';
    const ids=[...document.querySelectorAll('[data-lazy-model]')].map(n=>n.dataset.lazyModel);
    let complete=0;
    const result=await Promise.all(ids.map(async id=>{const ok=await quiet(id);complete++;status.textContent=`全文载入 / Loading full text: ${complete} / ${ids.length}`;return ok;}));
    const failed=result.filter(x=>!x).length;
    if(failed){allButton.disabled=false;allButton.textContent='重试未载入内容 / Retry remaining';status.textContent=`${failed} 个模型尚未载入，请重试。 / ${failed} model(s) still need to load.`;}
    else{allButton.textContent='全文已载入 / Full text ready';status.textContent='完整模型正文已载入，可以使用浏览器全文查找与打印。 / All model text is ready for browser find and print.';document.documentElement.dataset.fullText='ready';}
  });
  if('IntersectionObserver'in window){
    observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting)quiet(entry.target.dataset.lazyModel);},{rootMargin:'700px 0px'});
    document.querySelectorAll('[data-lazy-model]').forEach(body=>observer.observe(body));
  }
  function loadHash(){let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}if(isId(id))quiet(id);}
  window.addEventListener('hashchange',loadHash);loadHash();
  // Autoplay is an explicit reading control; it should not use CPU in a hidden tab.
  document.addEventListener('visibilitychange',()=>{
    if(!document.hidden)return;
    for(const id of Object.keys(APP_STATE.autoPlayTimers||{})){clearInterval(APP_STATE.autoPlayTimers[id]);delete APP_STATE.autoPlayTimers[id];const btn=document.querySelector('[data-play-id="'+id+'"]');if(btn){btn.textContent='AUTO';btn.classList.remove('playing');}}
  });
})();
