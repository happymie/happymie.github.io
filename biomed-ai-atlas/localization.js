/* One locale for the atlas, compendium and teaching studio. Text-only translation preserves original figures and formulas. */
(()=>{'use strict';
 const root=document.documentElement,BASE='/biomed-ai-atlas/',valid=l=>['zh','zh-Hant','en'].includes(l),normalize=s=>s.replace(/\s+/g,' ').trim();
 // Also release the gate left by older cached preference and CSS files.
 root.removeAttribute('data-locale-pending');
 let inferencePrepared=false;
 let locale='zh',ready=false,convert=s=>s,converter=null,observer,changing=false,scheduled=false,languageRequest=0;
 const originals=new WeakMap(),attributes=new WeakMap(),forward=new Map(),reverse=new Map(),pending=new Set();
 const skip='script,style,pre,code,.katex,math,#paper-citation,[translate="no"],.locale-selector';
 try{locale=localStorage.getItem('atlas.language')||'zh';}catch{}if(!valid(locale))locale='zh';
 const unknown=new Set(),traditionalSources=new Map();
 const originalTitle=document.title;
 const terms={"systematic scoping review": "系统性范围综述", "bilingual architectural atlas": "架构图谱", "cutoff": "检索截止", "interactive genealogy matrix": "交互谱系矩阵", "paradigm tracks": "范式分支", "interactive svg paradigm evolution constellation map": "交互范式演化星图", "all lineages": "全部谱系", "executive synthesis": "概要综述", "researcher competency taxonomy": "研究者能力分级", "three evolutionary axes": "三条演化主线", "three-layer system decomposition": "三层系统分解", "cross-paradigm landmarks": "跨范式里程碑", "clinical translation & evidence protocol": "临床转化与证据协议", "computational primitives": "计算基本单元", "dense prediction & self-configuring pipelines": "稠密预测与自配置流程", "pretraining objectives & task interfaces": "预训练目标与任务接口", "3d volumetric engineering & inverse problem ladder": "三维体积工程与逆问题阶梯", "registration, projective geometry, nerf/3dgs, gnn & temporal": "配准、投影几何、神经场、图与时序", "interactive clinical workflow decision protocol": "交互式临床流程决策", "evidence maturity & minimum evaluation matrix": "证据成熟度与最低评价矩阵", "chronological archive": "年代档案", "applicability boundaries & conclusions": "适用边界与结论", "interactive timeline": "交互演化年表", "evidence matrix": "证据矩阵", "records": "条记录", "entries": "条目", "total": "总计", "click column headers to sort": "点击列标题排序", "section": "章节", "family": "家族", "nodes": "节点", "stage": "阶段", "primary architectures": "核心架构", "paradigm families": "范式家族", "chinese–english parallel edition": "中文版本", "status": "状态", "all": "全部", "sort": "排序", "taxonomy order": "分类顺序", "spec": "条目", "year asc": "年份升序", "year desc": "年份降序", "name a": "名称 A", "tier": "层级", "input & physical measurement": "输入与物理测量", "source": "来源", "provenance": "来源记录","conceptual":"概念示意","long skips": "长跳连", "long skip": "长跳连", "self-attention": "自注意力", "cross-attention": "交叉注意力", "shifted-window": "移位窗口", "patch merging": "图像块合并", "patch embedding": "图像块嵌入", "image encoder": "图像编码器", "text encoder": "文本编码器", "tile encoder": "图块编码器", "slide encoder": "切片编码器", "depthwise": "逐通道", "pointwise": "逐点", "inverted bottleneck": "倒置瓶颈", "compound scaling": "复合缩放", "data consistency": "数据一致性", "data-consistency": "数据一致性", "dataset fingerprint": "数据集指纹", "foundation models": "基础模型", "foundation model": "基础模型", "encoder": "编码器", "decoder": "解码器", "concat": "拼接", "residual": "残差", "bottleneck": "瓶颈", "embedding": "嵌入", "feature": "特征", "features": "特征", "patches": "图像块", "patch": "图像块", "tokenizer": "标记器", "tokens": "标记", "token": "标记", "attention": "注意力", "mask": "掩码", "query": "查询", "queries": "查询", "keys": "键", "key": "键", "batch": "批次", "teacher": "教师", "student": "学生", "teacher–student": "教师与学生", "prompt": "提示", "prompts": "提示", "promptable segmentation": "提示式分割", "shared": "共享", "grid": "网格", "pooling": "池化", "head": "输出头", "gating": "门控", "gate": "门控", "skip": "跳连", "linear projection": "线性投影", "projection": "投影", "classification": "分类", "segmentation": "分割", "representation": "表征", "scan": "扫描", "selective": "选择性", "fusion": "融合", "supervoxel": "超体素", "pretraining": "预训练", "pretrained": "预训练", "contrastive": "对比", "benchmark": "基准", "logits": "类别分数", "conditioning": "条件控制", "score": "分数", "score field": "分数字段", "denoising": "去噪", "flow matching": "流匹配", "sampling": "采样", "unrolling": "展开", "unrolled": "展开式", "convolution": "卷积", "convolutions": "卷积", "Conv": "卷积", "dropout": "随机失活", "identity shortcut": "恒等跳连", "depth": "深度", "width": "宽度", "resolution": "分辨率", "spatial mixer": "空间混合器", "block": "模块", "blocks": "模块", "hierarchy": "层次结构", "tile": "图块", "slide": "切片", "instance": "实例", "displacement": "位移", "warping": "形变变换", "warp": "形变", "registration": "配准", "blind spot": "盲点", "clean image": "干净图像", "clean target": "干净目标", "noisy image": "含噪图像", "noisy": "含噪", "prior": "先验", "observation": "观测", "adversarial loss": "对抗损失", "adversarial supervision": "对抗监督", "supervision": "监督", "predictor": "预测器", "stop-gradient": "停止梯度", "momentum": "动量", "queue": "队列", "sharpening": "锐化", "centering": "中心化", "grounding": "定位", "caption": "图注", "captioning": "图像描述", "image–text": "图像与文本", "image-to-text": "图像到文本", "text-to-image": "文本到图像", "multi-head": "多头", "multi-scale": "多尺度", "global": "全局", "local": "局部", "reconstruction": "重建", "optimization": "优化", "Occlusion": "遮挡", "training": "训练", "inference": "推理", "train": "训练", "validation": "验证", "test": "测试", "source_route": "原始来源路径", "upstream": "上游代码", "paper_port": "论文方程移植", "mechanism_demo": "机制演示"};
 const termPattern=new RegExp('\\b('+Object.keys(terms).sort((a,b)=>b.length-a.length).map(s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')+')\\b','gi');
 const glossary=s=>s.replace(termPattern,m=>terms[m]||terms[Object.keys(terms).find(k=>k.toLowerCase()===m.toLowerCase())]||m);

 function localized(source,el){
  const key=normalize(source);if(!key)return source;
  if(el?.closest(skip))return source;
  const paired=el?.closest('.zh,.en,.lang-zh,.lang-en');
  // Hidden sibling editions retain their own source language, so switching is reversible.
  if(paired&&((locale==='en'&&paired.matches('.zh,.lang-zh'))||(locale!=='en'&&paired.matches('.en,.lang-en'))))return source;
  const canonical=traditionalSources.get(key)||key;let value=canonical;
  // Paired editions already carry reviewed copy. Only convert the Chinese
  // edition's script; dictionary synonyms must not rewrite navigation labels.
  if(paired?.matches('.zh,.en')){
   if(locale==='zh-Hant')value=convert(canonical);
   return source.slice(0,source.indexOf(source.trimStart()))+(locale==='en'?source.trim():value)+(/\s$/.test(source)?source.match(/\s+$/)[0]:'');
  }
  if(locale==='en'){
   value=forward.get(canonical)||canonical;
   // Home counters and model labels retain live values across all languages.
   value=value.split(/(\s+·\s+|\s+\/\s+)/).map(part=>forward.get(normalize(part))||part).join('');
   value=value.replace(/(\d+) 个模型、(\d+) 个家族。/g,'$1 models, $2 families.').replace(/(\d+) 个模型。/g,'$1 models.').replace(/(\d+) 个匹配模型 · 显示 (\d+) 个/g,'$1 matching models · Showing $2').replace(/再探索 (\d+) 个模型/g,'Explore $1 more models').replace(/(\d+) 个图谱模型/g,'$1 atlas models').replace(/窗口位置 (\d+) \/ (\d+)/g,'Window position $1 / $2').replace(/位置 (\d+)/g,'Position $1').replace(/权重之和 =/g,'Sum of weights =').replace(/ 原论文架构图/g,' original paper architecture').replace(/ 原论文图预览/g,' original figure preview').replace(/ 原论文图/g,' original paper figure').replace(/^查看 (.+) 的原图与详情$/,'View the original figure and details of $1').replace(/^放大 (.+) 原图$/,'Enlarge the original figure of $1').replace(/再探索 (\d+) 个模型/g,'Explore $1 more models');

   if(/^SOURCE:\s*/.test(value)){const tail=value.replace(/^SOURCE:\s*/,'');value='Source: '+(forward.get(tail)||tail);}
   if(/[\u3400-\u9fff]/.test(value)){
    const parts=value.split(/\s+(?:\/\/|\/|·)\s+/),english=parts.filter(s=>/[a-zA-Z]{2}/.test(s)&&!/[\u3400-\u9fff]/.test(s));
    if(english.length&&parts.some(s=>/[\u3400-\u9fff]/.test(s)))value=english.join(' · ');
   }
  }else{
   value=reverse.get(canonical)||reverse.get(canonical.toLowerCase())||(forward.has(canonical)?reverse.get(forward.get(canonical)):null)||canonical;
   value=value.split(/(\s*(?:\/\/|·)\s*)/).map(part=>reverse.get(normalize(part))||reverse.get(normalize(part).toLowerCase())||part).join('');
   if(!/[\u3400-\u9fff]/.test(value)&&(/^[^a-z\u3400-\u9fff]+$/.test(value)||/^(SECTION|FAMILY|MODEL|SPEC|STATUS|SORT|STAGE)\b/.test(value)||terms[value.toLowerCase()]))value=glossary(value);
   value=value.replace(/^([^:]+):/,(_,label)=>(reverse.get(label)||label)+':');
   value=value.replace(/(\d+) channels?/gi,'$1 个通道').replace(/(\d+) models?/gi,'$1 个模型').replace(/(\d+) READING NODES/gi,'$1 个阅读节点').replace(/EXPERIMENT \/ (\d+)/g,'实验 / $1').replace(/^CH (\d+)/,'通道 $1').replace(/^Layer (\d+)/,'第 $1 层');
   if(!/[\u3400-\u9fff]/.test(value)){
    const parts=value.split(/\s+(?:\/\/|\/)\s+/),chinese=parts.filter(s=>/[\u3400-\u9fff]/.test(s));if(chinese.length)value=chinese.join(' · ');
   }else if(!forward.has(key)){
    const parts=value.split(/\s+(?:\/\/|\/)\s+/);if(parts.length>1){const chinese=parts.filter(s=>/[\u3400-\u9fff]/.test(s));if(chinese.length)value=chinese.join(' · ');}
   }
   if(/[\u3400-\u9fff]/.test(value))value=glossary(value);
   if(locale==='zh-Hant'){const translated=convert(value);if(!traditionalSources.has(normalize(translated)))traditionalSources.set(normalize(translated),value);value=translated;}
  }
  if(locale==='en'&&/[\u3400-\u9fff]/.test(value))unknown.add(key);
  return source.slice(0,source.indexOf(source.trimStart()))+value+(/\s$/.test(source)?source.match(/\s+$/)[0]:'');
 }
 function text(node){const current=node.nodeValue,old=originals.get(node);const source=old&&current===old.last?old.source:current;const next=localized(source,node.parentElement);originals.set(node,{source,last:next});if(current!==next)node.nodeValue=next;}
 function attrs(el){if(el.matches?.(skip)||el.closest?.(skip))return;let stored=attributes.get(el);if(!stored){stored={};attributes.set(el,stored);}for(const name of ['aria-label','title','placeholder','alt']){const current=el.getAttribute?.(name);if(!current)continue;const old=stored[name],source=old&&current===old.last?old.source:current,next=localized(source,el);stored[name]={source,last:next};if(current!==next)el.setAttribute(name,next);}}
 function apply(target=document.body){if(!target||!ready)return;observer?.disconnect();const visit=node=>{if(node.nodeType===3){text(node);return;}if(node.nodeType!==1&&node.nodeType!==9)return;if(node.nodeType===1){if(node.matches(skip))return;attrs(node);}for(const child of node.childNodes)visit(child);};visit(target);observer?.observe(root,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['title','aria-label','alt','placeholder']});}
 function schedule(node){pending.add(node);if(scheduled)return;scheduled=true;queueMicrotask(()=>{scheduled=false;for(const n of pending)if(n.isConnected)apply(n);pending.clear();});}
 function controls(){document.querySelectorAll('.locale-selector').forEach(s=>{s.value=locale;s.setAttribute('aria-label',locale==='en'?'Language':locale==='zh-Hant'?'語言':'语言');});}
 function persist(){try{localStorage.setItem('atlas.language',locale);localStorage.setItem('biomed_atlas_lang_v3',locale==='en'?'en':'zh');localStorage.setItem('atlas_premium_lang',JSON.stringify(locale==='en'?'en':'zh'));localStorage.setItem('biomed_lab_lang',JSON.stringify(locale==='en'?'en':'zh'));}catch{}}
 async function traditional(){if(converter)return converter;converter=new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=BASE+'locales/traditional.js';const fail=()=>{converter=null;s.remove?.();reject(Error('Traditional Chinese resources could not load'));};s.onload=()=>{try{resolve(window.OpenCC.Converter({from:'cn',to:'t'}));}catch{fail();}};s.onerror=fail;document.head.append(s);});return converter;}
 async function setLanguage(next,{notify=true}={}){
  if(!valid(next))next='zh';const request=++languageRequest;changing=true;
  let conversion;try{conversion=next==='zh-Hant'?await traditional():convert;}catch(error){if(request===languageRequest){changing=false;controls();}throw error;}
  if(request!==languageRequest)return;
  convert=conversion;locale=next;
  root.dataset.lang=locale;root.lang=locale==='zh'?'zh-CN':locale==='zh-Hant'?'zh-Hant':'en';
  if(window.APP_STATE)window.APP_STATE.lang=locale==='en'?'en':'zh';persist();controls();document.title=localized(originalTitle);
  if(notify)document.dispatchEvent(new CustomEvent('atlas:language',{detail:{language:locale}}));
  root.dataset.lang=locale;root.lang=locale==='zh'?'zh-CN':locale==='zh-Hant'?'zh-Hant':'en';apply();changing=false;root.removeAttribute('data-locale-pending');
 }
 function selector(old){const s=document.createElement('select');s.className='locale-selector'+(old.classList.contains('language-button')?' language-button':'');s.setAttribute('aria-label','Language');s.setAttribute('translate','no');if(old.id)s.id=old.id;for(const [value,label]of [['zh','简体中文'],['zh-Hant','繁體中文'],['en','English']])s.append(new Option(label,value));s.value=locale;s.addEventListener('change',()=>setLanguage(s.value).catch(()=>{s.value=locale;const notice=document.getElementById('toast');if(notice){notice.textContent=locale==='en'?'Language resources could not load. Please try again.':locale==='zh-Hant'?'語言資源暫未載入，請重試。':'语言资源暂未加载，请重试。';notice.hidden=false;setTimeout(()=>notice.hidden=true,3500);}}));old.replaceWith(s);return s;}
 async function init(){
  const response=await fetch(BASE+'locales/dictionary.json');if(!response.ok)throw Error('Language resources unavailable');for(const [zh,en]of await response.json()){forward.set(normalize(zh),en);for(const key of [normalize(en),normalize(en).toLowerCase()]){const previous=reverse.get(key);if(!previous||(zh.match(/[A-Za-z]/g)||[]).length<(previous.match(/[A-Za-z]/g)||[]).length)reverse.set(key,zh);}}
  ready=true;
  for(const old of document.querySelectorAll('#language-toggle,#language,.language-button,.reference-nav select'))selector(old);
  // Legacy state is kept in sync without changing the retained chapter or image assets.
  window.setLanguageMode=mode=>setLanguage(mode);
  document.querySelectorAll('[data-lang-btn="bilingual"]').forEach(b=>b.remove());
  document.querySelectorAll('[data-lang-btn]').forEach(b=>{if(b.dataset.langBtn==='zh'){const t=b.cloneNode(true);t.dataset.langBtn='zh-Hant';t.removeAttribute('data-call-click');t.textContent='繁體中文';t.addEventListener('click',()=>setLanguage('zh-Hant'));b.after(t);}});
  // Enable lazy model controls only after the module has attached its handlers and prepared the sample.
  const inferenceControls=()=>['real-run','real-sample','real-file'].map(id=>document.getElementById(id)).filter(Boolean);
  document.addEventListener('click',event=>{if(!inferencePrepared&&event.target.closest('[data-studio-tab="inference"]'))inferenceControls().forEach(n=>n.disabled=true);},{capture:true});
  const statusNode=document.getElementById('real-status');
  if(statusNode&&/Sample ready|示例已就绪|示例已就緒/.test(statusNode.textContent))inferencePrepared=true;
  if(statusNode)new MutationObserver(()=>{if(inferencePrepared)return;const text=statusNode.textContent;if(/Sample ready|示例已就绪|示例已就緒|示例加载失败|示例加載失敗|Sample could not load/.test(text)){inferencePrepared=true;inferenceControls().forEach(n=>n.disabled=false);}}).observe(statusNode,{childList:true,subtree:true,characterData:true});
  observer=new MutationObserver(records=>{for(const r of records){if(r.type==='characterData')schedule(r.target);else if(r.type==='attributes')schedule(r.target);else for(const n of r.addedNodes)schedule(n);}});
  await setLanguage(locale);
  window.AtlasLocale={set:setLanguage,refresh:apply,t:(zh,en)=>locale==='en'?en:locale==='zh-Hant'?convert(glossary(zh)):glossary(zh),get language(){return locale;},untranslated:()=>[...unknown]};
  document.dispatchEvent(new Event('atlas:locale-ready'));
  window.addEventListener('storage',e=>{if(e.key==='atlas.language'&&valid(e.newValue))setLanguage(e.newValue);});
 }
 function loadSiteFeatures(){
  // A new navigation URL avoids reusing a cached previous personal edition.
  for(const anchor of document.querySelectorAll('a[href="/"]'))if(anchor.querySelector('img[src*="/brand/xiao-"]'))anchor.href='/?v=20261025-quiet08c';
  for(const path of ['serif-suite.css','fonts/atlas-song.css'])if(!document.querySelector('link[href$="'+path+'"]')){const fontStyle=document.createElement('link');fontStyle.rel='stylesheet';fontStyle.href=BASE+path;document.head.append(fontStyle);}const s=document.createElement('script');s.src=BASE+'site-features.js';s.defer=true;document.head.append(s);
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',loadSiteFeatures,{once:true});else loadSiteFeatures();
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>setTimeout(()=>init().catch(e=>{root.removeAttribute('data-locale-pending');console.error(e);}),0),{once:true});else setTimeout(()=>init().catch(e=>{root.removeAttribute('data-locale-pending');console.error(e);}),0);
})();
