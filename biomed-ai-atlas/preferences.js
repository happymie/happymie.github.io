(function(){
  try {
    var language=localStorage.getItem('atlas.language')||JSON.parse(localStorage.getItem('atlas_premium_lang')||'"zh"');
    var theme=localStorage.getItem('atlas.theme')||JSON.parse(localStorage.getItem('atlas_premium_appearance')||'"light"');
    document.documentElement.dataset.lang=language==='en'?'en':'zh';
    document.documentElement.lang=language==='en'?'en':'zh-CN';
    document.documentElement.dataset.theme=theme==='dark'?'dark':'light';
  } catch (_) {}
})();
