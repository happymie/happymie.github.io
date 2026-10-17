/* Apply a stored preference before the first paint; storage can be unavailable. */
(function(){
  'use strict';
  const html=document.documentElement;
  try {
    const parsed=key=>{try{return JSON.parse(localStorage.getItem(key));}catch{return null;}};
    const appearance=parsed('atlas_premium_appearance');
    if(['light','dark'].includes(appearance))html.dataset.appearance=appearance;
    if(location.pathname.includes('/teaching/')){
      const lang=parsed('biomed_lab_lang');if(['en','zh'].includes(lang))html.lang=lang==='en'?'en':'zh-CN';
    }else if(location.pathname.endsWith('/reference.html')){
      const lang=localStorage.getItem('biomed_atlas_lang_v3');if(['en','zh','bilingual'].includes(lang)){html.dataset.lang=lang;html.lang=lang==='en'?'en':'zh-CN';}
      const theme=localStorage.getItem('medai_theme');if(['light','dark','sepia'].includes(theme))html.dataset.theme=theme;
    }else{
      const lang=parsed('atlas_premium_lang');if(['en','zh','bilingual'].includes(lang)){html.dataset.lang=lang;html.lang=lang==='en'?'en':'zh-CN';}
    }
  }catch{/* Private browsing still has a usable default. */}
})();
