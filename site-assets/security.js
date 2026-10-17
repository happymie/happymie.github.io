/* Shared, dependency-free controls for trusted site code and untrusted content. */
(function (root) {
  'use strict';
  const allowed = new Set(('showConstellationTooltip hideConstellationTooltip toggleSidebarToc switchWorkspace setLanguageMode openGlobalSearchModal openCompareModal cycleTheme toggleFontMode filterTocItems jumpToSection jumpToModelCard toggleTocFamily filterOnlyFamily jumpToModelByName filterTimelineYear setFamilyFilter toggleCollapseFamilySection selectWizardWorkflow filterConstellationLineage selectPipelineStage loadPresetComparison resetAllModelFilters setModelViewMode sortInteractiveTable exportTableCSV filterInteractiveTable openSourceModal openLightbox expandSingleCardModal toggleCompareModel toggleBookmarkModel toggleMasteredModel copyModelSummary selectArchNode stepArchNode toggleAutoPlayNodes onModelSearchInput onEraFilterChange onVenueFilterChange onSpecialFilterChange onSortOrderChange zoomLightbox closeLightbox clearCompareList closeCompareModal closeQuickViewModal closeSourceModal closeDeployModal closeCitationModal closeGlobalSearchModal openDeployModal openCitationModal copyDeployCode onGlobalPaletteSearch renderCompareModalBody copyElementText scrollToTop').split(' '));

  // This parses a deliberately small data language. It never evaluates JavaScript.
  function parseCalls(source) {
    if (typeof source !== 'string' || source.length > 4096) throw Error('Invalid action');
    let p = 0, self = false, prevent = false;
    const calls = [], ws = () => { while (/\s/.test(source[p] || '') && p < source.length) p++; };
    const expect = ch => { ws(); if (source[p++] !== ch) throw Error('Invalid action syntax'); };
    function string() {
      const quote = source[p++]; let value = '';
      while (p < source.length) {
        let c = source[p++];
        if (c === quote) return value;
        if (c === '\\') {
          c = source[p++];
          const escapes = {n:'\n',r:'\r',t:'\t',b:'\b',f:'\f',"'":"'",'"':'"','\\':'\\','/':'/'};
          if (c === 'u') {
            const hex = source.slice(p,p+4); if (!/^[\da-f]{4}$/i.test(hex)) throw Error('Invalid escape');
            value += String.fromCharCode(parseInt(hex,16)); p += 4;
          } else if (Object.hasOwn(escapes,c)) value += escapes[c];
          else throw Error('Invalid escape');
        } else { if (c < ' ') throw Error('Invalid character'); value += c; }
      }
      throw Error('Unclosed string');
    }
    function value(depth = 0) {
      ws(); if (depth > 3) throw Error('Action is too deep');
      const ch = source[p]; if (ch === "'" || ch === '"') return string();
      if (ch === '[') {
        p++; const items=[]; ws();
        while(source[p] !== ']') { if(items.length >= 32) throw Error('Too many arguments'); items.push(value(depth+1)); ws(); if(source[p] !== ',') break; p++; }
        expect(']'); return items;
      }
      const number = source.slice(p).match(/^-?(?:\d+(?:\.\d+)?|\.\d+)/);
      if (number) {p += number[0].length; const n=Number(number[0]); if(!Number.isFinite(n)) throw Error('Invalid number'); return n;}
      for(const ref of ['this.value','event','this']) if(source.startsWith(ref,p)){p+=ref.length;return {$ref:ref};}
      for(const [literal,v] of [['true',true],['false',false],['null',null]]) if(source.startsWith(literal,p)){p+=literal.length;return v;}
      throw Error('Only literal arguments and control values are allowed');
    }
    ws(); const condition=source.slice(p).match(/^if\s*\(\s*event\.target\s*===\s*this\s*\)/);
    if(condition){self=true;p+=condition[0].length;}
    while(p < source.length) {
      ws(); if(p >= source.length) break;
      const done=source.slice(p).match(/^return\s+false\s*;?\s*$/);
      if(done){prevent=true;p=source.length;break;}
      const name=source.slice(p).match(/^[A-Za-z]\w*/)?.[0];
      if(!name || !allowed.has(name) || calls.length >= 8) throw Error('Unapproved action');
      p+=name.length;expect('('); const args=[];ws();
      while(source[p] !== ')'){if(args.length >= 8)throw Error('Too many arguments');args.push(value());ws();if(source[p]!==',')break;p++;}
      expect(')');calls.push({name,args});ws();if(p<source.length)expect(';');
    }
    if(!calls.length)throw Error('Empty action');
    return {calls,self,prevent};
  }
  function safeURL(value, {asset=false} = {}) {
    try {
      if(typeof value !== 'string' || !value.trim() || /[\u0000-\u001f\u007f]/.test(value)) return '#';
      const base = root.location?.href || 'https://happymie.github.io/';
      const url = new URL(value,base), origin = new URL(base).origin;
      if(url.username || url.password || !['https:','http:'].includes(url.protocol)) return '#';
      if(url.protocol==='http:' && url.origin!==origin) return '#';
      if(asset && url.origin!==origin) return '#';
      return url.href;
    } catch {return '#';}
  }
  const escapeHTML = value => String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function fragment(html) {
    if(typeof html !== 'string' || html.length > 600000) throw Error('Invalid model document');
    const template=document.createElement('template');template.innerHTML=html;
    template.content.querySelectorAll('script,style,iframe,object,embed,base,link,meta,form,foreignObject,animate,animateTransform,set').forEach(n=>n.remove());
    for(const el of template.content.querySelectorAll('*')) {
      for(const attr of [...el.attributes]) {
        const key=attr.name.toLowerCase(),value=attr.value;
        if(key.startsWith('on') || ['srcdoc','formaction','action','srcset','ping'].includes(key)){el.removeAttribute(attr.name);continue;}
        if(key.startsWith('data-call-')){try{parseCalls(value);}catch{el.removeAttribute(attr.name);}continue;}
        if(['href','src','xlink:href'].includes(key)) {
          if(value.startsWith('#')) continue;
          if(key==='href' && el.tagName.toLowerCase()==='a') el.setAttribute(attr.name,safeURL(value));
          else el.setAttribute(attr.name,safeURL(value,{asset:true}));
        }
      }
      if(el.tagName.toLowerCase()==='a' && el.getAttribute('target')==='_blank')el.setAttribute('rel','noopener noreferrer');
    }
    return template.content;
  }
  const api=Object.freeze({parseCalls,safeURL,escapeHTML,fragment});
  if(typeof module==='object' && module.exports)module.exports=api;
  root.SiteSecurity=api;
  if(typeof document==='undefined')return;
  root.copyElementText=(id,message)=>root.copyTextSafe?.(document.getElementById(id)?.textContent||'',message);
  root.scrollToTop=()=>root.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
  const cache = new Map();
  const resolve=(v,el,event)=>Array.isArray(v)?v.map(x=>resolve(x,el,event)):v&&typeof v==='object'&&v.$ref?(v.$ref==='this.value'?el.value:v.$ref==='this'?el:event):v;
  function dispatch(event,type) {
    for(const el of event.composedPath()) {
      if(!(el instanceof Element))continue;
      const source=el.getAttribute('data-call-'+type);if(!source)continue;
      if((type==='mouseenter'||type==='mouseleave') && event.relatedTarget instanceof Node && el.contains(event.relatedTarget))continue;
      try {
        let parsed=cache.get(source);if(!parsed){parsed=parseCalls(source);if(cache.size>2048)cache.clear();cache.set(source,parsed);}
        if(parsed.self && event.target!==el)continue;
        if(parsed.prevent)event.preventDefault();
        for(const call of parsed.calls){const fn=root[call.name];if(typeof fn==='function')fn.apply(el,call.args.map(v=>resolve(v,el,event)));}
      } catch(error){console.warn('Ignored invalid site action:',error.message);}
      if(event.cancelBubble)break;
    }
  }
  for(const type of ['click','input','change'])document.addEventListener(type,event=>dispatch(event,type));
  document.addEventListener('mouseover',event=>dispatch(event,'mouseenter'));
  document.addEventListener('mouseout',event=>dispatch(event,'mouseleave'));
  document.addEventListener('keydown',event=>{
    const el=event.target;
    if(!(el instanceof Element)||!el.matches('[data-call-click][role="button"]')||el.matches('button,a,input,select,textarea'))return;
    if(event.key==='Enter'||event.key===' '){event.preventDefault();el.click();}
  });
})(typeof window==='undefined'?globalThis:window);
