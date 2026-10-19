(function(){
  document.documentElement.removeAttribute('data-locale-pending');
  try {
    var language=localStorage.getItem('atlas.language')||JSON.parse(localStorage.getItem('atlas_premium_lang')||'"zh"');
    var theme=localStorage.getItem('atlas.theme')||JSON.parse(localStorage.getItem('atlas_premium_appearance')||'"light"');
    document.documentElement.dataset.lang=['en','zh-Hant'].includes(language)?language:'zh';
    document.documentElement.lang=language==='en'?'en':language==='zh-Hant'?'zh-Hant':'zh-CN';
    document.documentElement.dataset.theme=theme==='dark'?'dark':'light';
  } catch (_) {}
})();
