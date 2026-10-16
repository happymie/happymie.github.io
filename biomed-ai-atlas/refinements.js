/* Reading controls and accessible overlays. Original atlas behavior is retained. */
(function(){
 'use strict';
 const $=s=>document.querySelector(s), $$=s=>Array.from(document.querySelectorAll(s));
 const read=(k)=>{try{return localStorage.getItem(k)}catch(e){return null}};
 const write=(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}};
 const mobile=()=>window.matchMedia('(max-width:1120px)').matches;
 const scrim=$('.atlas-menu-scrim'),sidebar=$('#sidebar-toc');
 const originalToggle=window.toggleSidebarToc;
 function closeIndex(){sidebar.classList.remove('mobile-open');scrim.classList.remove('visible');$('.toc-drawer-btn').setAttribute('aria-expanded','false')}
 window.toggleSidebarToc=function(){if(mobile()){sidebar.classList.remove('collapsed');const open=sidebar.classList.toggle('mobile-open');scrim.classList.toggle('visible',open);$('.toc-drawer-btn').setAttribute('aria-expanded',String(open));if(open)$('#toc-filter-input').focus()}else{originalToggle();$('.toc-drawer-btn').setAttribute('aria-expanded',String(!sidebar.classList.contains('collapsed')))}};
 matchMedia('(max-width:1120px)').addEventListener('change',closeIndex);scrim.addEventListener('click',closeIndex);sidebar.addEventListener('click',e=>{if(mobile()&&e.target.closest('a'))closeIndex()});
 const closeIndexButton=document.createElement('button');closeIndexButton.className='btn-hairline mobile-index-close';closeIndexButton.textContent='关闭 / Close';closeIndexButton.onclick=closeIndex;sidebar.querySelector('.toc-header').prepend(closeIndexButton);
 const indexStyle=document.createElement('style');indexStyle.textContent='.mobile-index-close{display:none}@media(max-width:1120px){.mobile-index-close{display:flex;margin-bottom:16px}}';document.head.append(indexStyle);
 $('.toc-drawer-btn').setAttribute('aria-controls','sidebar-toc');$('.toc-drawer-btn').setAttribute('aria-expanded',String(!mobile()));
 const location=document.createElement('div');location.className='reading-location';location.textContent='READ · EXPLORE · UNDERSTAND';sidebar.append(location);
 const originalJump=window.jumpToModelCard;
 window.jumpToModelCard=function(id){originalJump(id);write('atlas_last_model',id);if(mobile())closeIndex();const card=document.getElementById(id);if(card){card.setAttribute('tabindex','-1');setTimeout(()=>card.focus({preventScroll:true}),400)}};
 const last=read('atlas_last_model');const model=(window.MODELS_META||[]).find(m=>m.id===last);
 if(model){$('#resume-strip').classList.add('visible');$('#resume-model').textContent=model.title+' →';$('#resume-model').onclick=()=>jumpToModelCard(model.id)}
 $$('[data-hero-model]').forEach(el=>{const go=()=>{const q=el.dataset.heroModel.toLowerCase();const m=(window.MODELS_META||[]).find(x=>x.title.toLowerCase()===q)||(window.MODELS_META||[]).find(x=>x.title.toLowerCase().includes(q));if(m)jumpToModelCard(m.id)};el.addEventListener('click',go);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();go()}})});
 const lang=read('biomed_atlas_lang_v3');if(['zh','en','bilingual'].includes(lang)){document.documentElement.dataset.lang=lang;APP_STATE.lang=lang;$$('.lang-seg-btn').forEach(b=>b.classList.toggle('active',b.dataset.langBtn===lang))}
 const originalLang=window.setLanguageMode;window.setLanguageMode=function(mode){originalLang(mode);document.documentElement.lang=mode==='en'?'en':'zh-CN';$$('.lang-seg-btn').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.langBtn===mode)))};
 document.documentElement.lang=lang==='en'?'en':'zh-CN';$$('.lang-seg-btn').forEach(b=>b.setAttribute('aria-pressed',String(b.classList.contains('active'))));
 const originalWorkspace=window.switchWorkspace;window.switchWorkspace=function(mode){originalWorkspace(mode);$$('.ws-tab-btn').forEach(b=>b.setAttribute('aria-current',b.dataset.wsTab===mode?'page':'false'));if(mobile())closeIndex()};
 // Overlay focus handling works for every existing dialog, including deployment help.
 const modals=$$('.modal-backdrop');let priorFocus=null,active=null;
 function syncDialogs(){const open=modals.filter(m=>m.classList.contains('open'));const next=open[open.length-1]||null;if(next!==active){if(next){if(!active)priorFocus=document.activeElement;active=next;const f=next.querySelector('input,button,a[href],[tabindex="0"]');if(f)f.focus()}else{active=null;if(priorFocus&&priorFocus.isConnected)priorFocus.focus();priorFocus=null}}document.body.style.overflow=next?'hidden':''}
 modals.forEach(m=>{m.setAttribute('role','dialog');m.setAttribute('aria-modal','true');const title=m.querySelector('.modal-title');if(title){if(!title.id)title.id=m.id+'-accessible-title';m.setAttribute('aria-labelledby',title.id)}const close=m.querySelector('.modal-close-btn');if(close)close.setAttribute('aria-label','Close dialog / 关闭');new MutationObserver(syncDialogs).observe(m,{attributes:true,attributeFilter:['class']});m.addEventListener('click',e=>{if(e.target===m)m.classList.remove('open')})});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeIndex();modals.forEach(m=>m.classList.remove('open'))}if(e.key==='Tab'&&active){const els=Array.from(active.querySelectorAll('button,a[href],input,select,textarea,[tabindex="0"]')).filter(el=>el.getClientRects().length&&!el.disabled);if(!els.length)return;const first=els[0],last=els[els.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}}});
 const search=$('#global-palette-input');if(search)search.addEventListener('keydown',e=>{if(e.key==='Enter'){const row=$('#global-palette-results [onclick]');if(row){e.preventDefault();row.click()}}});
 $$('.arch-figure-img,.gallery-thumb-img').forEach(i=>{i.loading='lazy';i.decoding='async'});
 if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){const c=entry.target;const title=c.querySelector('.spec-model-title');if(title){location.textContent='READING / '+title.textContent.trim();$$('.toc-submodel-link').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+c.id))}}})},{rootMargin:'-110px 0px -60% 0px',threshold:0});$$('.model-card').forEach(c=>observer.observe(c))}
 $$('input').forEach(i=>{if(!i.getAttribute('aria-label')&&!i.labels?.length)i.setAttribute('aria-label',i.placeholder||'Filter models')});
})();



{const s=document.createElement('script');s.src='reference-upgrade.js?v=2.1.2';document.body.appendChild(s)}
