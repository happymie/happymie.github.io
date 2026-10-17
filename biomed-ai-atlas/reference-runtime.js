function flattenSearchValues(value) { return value && typeof value === 'object' ? Object.values(value).flatMap(flattenSearchValues) : [String(value == null ? '' : value)]; }
var APP_STATE = {
  theme: 'light',
  font: 'modern',
  workspace: 'all',
  viewMode: 'cards',
  familyFilter: 'all',
  eraFilter: 'all',
  venueFilter: 'all',
  specialFilter: 'all',
  sortOrder: 'book',
  searchQuery: '',
  compareList: [],
  bookmarks: {},
  mastered: {},
  nodeStepState: {},
  autoPlayTimers: {},
  lightboxModelId: null,
  lightboxScale: 1,
  lightboxTx: 0,
  lightboxTy: 0,
  isDraggingLb: false,
  dragStartX: 0,
  dragStartY: 0
};

var MODELS_MAP = {};
var SOURCES_MAP = {};
var WORKFLOWS_DATA = [];

function safeLoadStorage() {
  try {
    var savedTheme = localStorage.getItem('medai_theme');
    if (savedTheme) APP_STATE.theme = savedTheme;
    var savedFont = localStorage.getItem('medai_font');
    if (savedFont) APP_STATE.font = savedFont;
    var savedBm = localStorage.getItem('medai_bookmarks');
    if (savedBm) APP_STATE.bookmarks = JSON.parse(savedBm) || {};
    var savedMs = localStorage.getItem('medai_mastered');
    if (savedMs) APP_STATE.mastered = JSON.parse(savedMs) || {};
  } catch (e) {}
}

function safeSaveStorage() {
  try {
    localStorage.setItem('medai_theme', APP_STATE.theme);
    localStorage.setItem('medai_font', APP_STATE.font);
    localStorage.setItem('medai_bookmarks', JSON.stringify(APP_STATE.bookmarks));
    localStorage.setItem('medai_mastered', JSON.stringify(APP_STATE.mastered));
  } catch (e) {}
}

function showToast(msg) {
  var container = document.getElementById('toast-container');
  if (!container) return;
  var el = document.createElement('div');
  el.className = 'toast-msg';
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(function() {
    el.style.opacity = '0';
    el.style.transition = 'opacity 0.22s ease';
    setTimeout(function() {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, 240);
  }, 2200);
}

function copyTextSafe(text, successMsg) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function() {
      showToast(successMsg || 'COPIED TO CLIPBOARD');
    }).catch(function() {
      fallbackCopyText(text, successMsg);
    });
  } else {
    fallbackCopyText(text, successMsg);
  }
}

function fallbackCopyText(text, successMsg) {
  var ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.left = '-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
    showToast(successMsg || 'COPIED TO CLIPBOARD');
  } catch (e) {
    showToast('COPY FAILED');
  }
  document.body.removeChild(ta);
}

function cycleTheme() {
  var order = ['light', 'dark', 'sepia'];
  var nextIdx = (order.indexOf(APP_STATE.theme) + 1) % order.length;
  setTheme(order[nextIdx]);
}

function setTheme(themeName) {
  APP_STATE.theme = themeName;
  document.documentElement.setAttribute('data-theme', themeName);
  var btn = document.getElementById('theme-toggle-btn');
  if (btn) {
    if (themeName === 'light') btn.innerHTML = '<span>PAPER</span>';
    else if (themeName === 'dark') btn.innerHTML = '<span>OBSIDIAN</span>';
    else btn.innerHTML = '<span>DAYLIGHT</span>';
  }
  safeSaveStorage();
}

function toggleFontMode() {
  var nextFont = APP_STATE.font === 'modern' ? 'academic' : 'modern';
  APP_STATE.font = nextFont;
  document.documentElement.setAttribute('data-font', nextFont);
  var btn = document.getElementById('font-toggle-btn');
  if (btn) {
    btn.innerHTML = nextFont === 'modern' ? '<span>SANS+SERIF</span>' : '<span>SERIF ONLY</span>';
  }
  safeSaveStorage();
  showToast(nextFont === 'modern' ? 'TYPOGRAPHY: SWISS SANS + EDITORIAL SERIF' : 'TYPOGRAPHY: CLASSIC ACADEMIC SERIF');
}

function toggleSidebarToc() {
  var sb = document.getElementById('sidebar-toc');
  if (!sb) return;
  sb.classList.toggle('collapsed');
}

function switchWorkspace(mode) {
  APP_STATE.workspace = mode;
  var tabs = document.querySelectorAll('.ws-tab-btn');
  for (var i = 0; i < tabs.length; i++) {
    tabs[i].classList.toggle('active', tabs[i].getAttribute('data-ws-tab') === mode);
  }
  var secConst = document.getElementById('section-constellation-map');
  var secReview = document.getElementById('workspace-review-part');
  var secWizard = document.getElementById('section-decision-wizard');
  var secTimeline = document.getElementById('section-timeline');
  var secTables = document.getElementById('section-all-tables-hub');
  var secModels = document.getElementById('section-models-compendium');
  var secConclusions = document.getElementById('section-conclusions-part');

  if (mode === 'all') {
    if (secConst) secConst.style.display = '';
    if (secReview) secReview.style.display = '';
    if (secWizard) secWizard.style.display = '';
    if (secTimeline) secTimeline.style.display = '';
    if (secTables) secTables.style.display = '';
    if (secModels) secModels.style.display = '';
    if (secConclusions) secConclusions.style.display = '';
  } else if (mode === 'explorer') {
    if (secConst) secConst.style.display = 'none';
    if (secReview) secReview.style.display = 'none';
    if (secWizard) secWizard.style.display = 'none';
    if (secTimeline) secTimeline.style.display = 'none';
    if (secTables) secTables.style.display = 'none';
    if (secModels) secModels.style.display = '';
    if (secConclusions) secConclusions.style.display = 'none';
    if (secModels) secModels.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else if (mode === 'timeline') {
    if (secConst) secConst.style.display = '';
    if (secReview) secReview.style.display = 'none';
    if (secWizard) secWizard.style.display = 'none';
    if (secTimeline) secTimeline.style.display = '';
    if (secTables) secTables.style.display = 'none';
    if (secModels) secModels.style.display = 'none';
    if (secConclusions) secConclusions.style.display = 'none';
    if (secConst) secConst.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else if (mode === 'wizard') {
    if (secConst) secConst.style.display = 'none';
    if (secReview) secReview.style.display = 'none';
    if (secWizard) secWizard.style.display = '';
    if (secTimeline) secTimeline.style.display = 'none';
    if (secTables) secTables.style.display = 'none';
    if (secModels) secModels.style.display = 'none';
    if (secConclusions) secConclusions.style.display = 'none';
    if (secWizard) secWizard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else if (mode === 'tables') {
    if (secConst) secConst.style.display = 'none';
    if (secReview) secReview.style.display = 'none';
    if (secWizard) secWizard.style.display = 'none';
    if (secTimeline) secTimeline.style.display = 'none';
    if (secTables) secTables.style.display = '';
    if (secModels) secModels.style.display = 'none';
    if (secConclusions) secConclusions.style.display = '';
    if (secTables) secTables.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function jumpToSection(secId) {
  if (APP_STATE.workspace !== 'all') {
    switchWorkspace('all');
  }
  var el = document.getElementById(secId);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function jumpToModelCard(modelId) {
  hideConstellationTooltip();
  if (APP_STATE.workspace !== 'all' && APP_STATE.workspace !== 'explorer') {
    switchWorkspace('all');
  }
  var card = document.getElementById(modelId);
  if (card && card.style.display === 'none') {
    resetAllModelFilters();
  }
  if (APP_STATE.viewMode === 'table') {
    setModelViewMode('cards');
  }
  if (card) {
    card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    card.style.outline = '2px solid var(--accent-secondary)';
    setTimeout(function() {
      card.style.outline = '';
    }, 2200);
  }
}

function jumpToModelByName(nameQuery) {
  var q = String(nameQuery || '').toLowerCase().split('/')[0].trim();
  for (var i = 0; i < window.MODELS_META.length; i++) {
    var m = window.MODELS_META[i];
    if (m.title.toLowerCase().indexOf(q) !== -1 || q.indexOf(m.title.toLowerCase()) !== -1) {
      jumpToModelCard(m.id);
      return;
    }
  }
  switchWorkspace('explorer');
  var inp = document.getElementById('model-search-input');
  if (inp) {
    inp.value = q;
    onModelSearchInput(q);
  }
}

// ============================================================================
// Interactive SVG Constellation Map Handlers
// ============================================================================
function filterConstellationLineage(linKey, btnEl) {
  var btns = document.querySelectorAll('.lineage-filter-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove('active');
  }
  if (btnEl) btnEl.classList.add('active');

  var paths = document.querySelectorAll('.const-lineage-path');
  for (var j = 0; j < paths.length; j++) {
    var p = paths[j];
    var pk = p.getAttribute('data-lineage');
    if (linKey === 'all') {
      p.style.strokeOpacity = '0.45';
      p.style.strokeWidth = '1.75';
    } else if (pk === linKey) {
      p.style.strokeOpacity = '0.95';
      p.style.strokeWidth = '3';
    } else {
      p.style.strokeOpacity = '0.08';
      p.style.strokeWidth = '1';
    }
  }
}

function showConstellationTooltip(evt, modelId) {
  var m = MODELS_MAP[modelId];
  var tip = document.getElementById('constellation-tooltip');
  if (!m || !tip) return;
  var card = document.getElementById(modelId);
  var imgEl = card ? card.querySelector('.arch-figure-img') : null;
  var imgSrc = imgEl ? imgEl.src : '';

  tip.innerHTML =
    '<div style="font-family:var(--font-mono);font-size:10px;color:var(--accent-secondary);letter-spacing:0.06em;margin-bottom:4px">' +
      'SPEC ' + m.num_str + ' / 137 · ' + m.year_str + ' · ' + m.venue +
    '</div>' +
    '<div style="font-family:var(--font-serif);font-size:18px;font-weight:700;margin-bottom:6px">' + m.title + '</div>' +
    (imgSrc ? '<div style="background:#FFF;border:1px solid var(--rule);padding:6px;margin-bottom:8px;text-align:center"><img src="' + imgSrc + '" alt="" style="max-width:100%;max-height:95px;object-fit:contain" /></div>' : '') +
    '<div class="lang-zh" style="font-size:12px;line-height:1.45;color:var(--ink-secondary)">' + m.core_mech + '</div>' +
    '<div class="lang-en" style="font-size:11.5px;line-height:1.45;color:var(--ink-tertiary);margin-top:3px">' + (m.core_mech_en || '') + '</div>' +
    '<div style="margin-top:6px;font-family:var(--font-mono);font-size:9.5px;color:var(--ink-tertiary)">CLICK NODE TO INSPECT SPECIFICATION ↗</div>';

  tip.style.display = 'block';
  var x = evt.clientX + 16;
  var y = evt.clientY - 20;
  if (x + 320 > window.innerWidth) x = evt.clientX - 326;
  if (y + 240 > window.innerHeight) y = window.innerHeight - 250;
  if (y < 12) y = 12;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}

function hideConstellationTooltip() {
  var tip = document.getElementById('constellation-tooltip');
  if (tip) tip.style.display = 'none';
}

var PIPELINE_STAGES = [
  {
    title: 'STAGE 01 // 原始物理测量与输入 (Input & Physical Measurement)',
    desc: '临床原始信号既包括重建后的图像网格（CT HU、MRI 强度、WSI 图像块、导联波形），也包括 k-space、sinogram 投影、PET 符合计数与相机位姿参数 φ。任何架构设计必须先明确输入的数据单位（patient / study / volume / series / slice / patch）与物理层厚（anisotropic spacing）。'
  },
  {
    title: 'STAGE 02 // 表示计算骨干 (Representation Backbone)',
    desc: '局部共享卷积（ResNet / ConvNeXt / MedNeXt）提供平移等变与小样本归纳偏置；Self-Attention（ViT / Swin）建立输入相关的动态 token 交互；Selective SSM（Mamba / VMamba）将长序列历史压入有限状态。三者改变的是局部与全局证据的聚合复杂度，而混合结构常最符合医学体数据约束。'
  },
  {
    title: 'STAGE 03 // 监督来源与任务接口 (Supervision & Task Interface)',
    desc: '人工标注之外，对比学习（SimCLR / MoCo）、遮挡重建（MAE）、自蒸馏（DINOv2）与图文对齐（CLIP / BioViL）将复用单位扩展为通用编码器；U-Net 长跳连、SAM 点框提示与 VLM 连接器（Q-Former / LLaVA）定义了特征如何转化为密集掩膜或结构化报告草稿。'
  },
  {
    title: 'STAGE 04 // 物理算子与几何可微闭环 (Physics & Geometry Consistency)',
    desc: '在逆问题重建（MoDL / VarNet / DPS）与 2D/3D 配准（VoxelMorph / GradICON / DiffPose）中，把前向投影算子 A、伴随算子 A*、微分同胚变形场或可微 DRR 嵌入计算图，显式区分测量空间残差与先验补全，防范零空间（null-space）解剖幻觉。'
  },
  {
    title: 'STAGE 05 // 域外校准、不确定性与人工复核 (Calibration & Audit Fallback)',
    desc: '模型输出进入临床工作流前，必须经过概率校准、分布外（OOD）检测、最坏情况（worst-case / HD95 / folding）审计与安全回退机制（如配准残差超限退回人工初始化、VLM 报告提供视觉 grounding 与医生签字确认）。'
  }
];

function selectPipelineStage(idx) {
  var btns = document.querySelectorAll('.pipe-step-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.toggle('active', i === idx);
  }
  var banner = document.getElementById('pipe-stage-detail');
  if (banner && PIPELINE_STAGES[idx]) {
    banner.innerHTML = '<strong style="color:var(--ink-primary)">' + PIPELINE_STAGES[idx].title + '：</strong>' + PIPELINE_STAGES[idx].desc;
  }
}

function selectWizardWorkflow(idx) {
  var btns = document.querySelectorAll('.wiz-tab-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.toggle('active', i === idx);
  }
  var row = WORKFLOWS_DATA[idx];
  var panel = document.getElementById('wizard-detail-panel');
  if (!panel || !row) return;

  var keywords = (row.baseline + ' ' + row.advanced).split(/[\/、，,\s()（）+]+/);
  var matchedModels = [];
  for (var mIdx = 0; mIdx < window.MODELS_META.length; mIdx++) {
    var m = window.MODELS_META[mIdx];
    for (var kIdx = 0; kIdx < keywords.length; kIdx++) {
      var kw = keywords[kIdx].trim().toLowerCase();
      if (kw.length >= 3 && m.title.toLowerCase().indexOf(kw) !== -1) {
        matchedModels.push(m);
        break;
      }
    }
  }

  var chipsHtml = matchedModels.slice(0, 8).map(function(m) {
    return '<button type="button" class="trunk-node-btn" data-call-click="expandSingleCardModal(\'' + m.id + '\')">' +
           '<span>SPEC ' + m.num_str + ' · ' + m.title + ' (' + m.year + ') ↗</span></button>';
  }).join('');

  panel.innerHTML =
    '<div class="mono-kicker">WORKFLOW DECISION PROTOCOL // 0' + row.order + '</div>' +
    '<h3 style="margin:4px 0 12px;font-size:22px;font-family:var(--font-serif)">' + row.workflow + '</h3>' +
    '<div class="wiz-cards-grid">' +
      '<div class="wiz-box"><span class="wiz-box-label">01 // VERIFIED STRONG BASELINE (先做强基线)</span><strong>' + row.baseline + '</strong></div>' +
      '<div class="wiz-box"><span class="wiz-box-label">02 // FRONTIER MECHANISMS (再考虑新机制)</span><span>' + row.advanced + '</span></div>' +
      '<div class="wiz-box"><span class="wiz-box-label">03 // MANDATORY EVALUATION (必须评价指标)</span><span>' + row.evaluation + '</span></div>' +
      '<div class="wiz-box"><span class="wiz-box-label">04 // CLINICAL SAFETY FALLBACK (失败安全回退)</span><span>' + row.fallback + '</span></div>' +
    '</div>' +
    (chipsHtml ? '<div style="margin-top:18px"><div class="mono-kicker" style="margin-bottom:8px">MATCHED ARCHITECTURES IN COMPENDIUM:</div><div class="trunk-nodes-flow">' + chipsHtml + '</div></div>' : '');
}

function filterInteractiveTable(tableId, query) {
  var tbl = document.getElementById(tableId);
  if (!tbl || !tbl.tBodies[0]) return;
  var q = String(query || '').trim().toLowerCase();
  var rows = tbl.tBodies[0].rows;
  for (var i = 0; i < rows.length; i++) {
    var txt = rows[i].textContent.toLowerCase();
    rows[i].style.display = (!q || txt.indexOf(q) !== -1) ? '' : 'none';
  }
}

var TABLE_SORT_DIRS = {};
function sortInteractiveTable(tableId, colIdx, colType) {
  var tbl = document.getElementById(tableId);
  if (!tbl || !tbl.tBodies[0]) return;
  var key = tableId + '_' + colIdx;
  var asc = !TABLE_SORT_DIRS[key];
  TABLE_SORT_DIRS[key] = asc;

  var tbody = tbl.tBodies[0];
  var rows = Array.prototype.slice.call(tbody.rows);
  rows.sort(function(a, b) {
    var cellA = a.cells[colIdx];
    var cellB = b.cells[colIdx];
    var valA = cellA ? (cellA.getAttribute('data-sort') || cellA.textContent.trim()) : '';
    var valB = cellB ? (cellB.getAttribute('data-sort') || cellB.textContent.trim()) : '';
    if (colType === 'number') {
      var numA = parseFloat(String(valA).replace(/,/g, '')) || 0;
      var numB = parseFloat(String(valB).replace(/,/g, '')) || 0;
      return asc ? (numA - numB) : (numB - numA);
    }
    return asc ? String(valA).localeCompare(String(valB), 'zh-CN') : String(valB).localeCompare(String(valA), 'zh-CN');
  });
  for (var i = 0; i < rows.length; i++) {
    tbody.appendChild(rows[i]);
  }
}

function exportTableCSV(tableId, title) {
  var tbl = document.getElementById(tableId);
  if (!tbl) return;
  var csvRows = [];
  var allRows = tbl.querySelectorAll('tr');
  for (var i = 0; i < allRows.length; i++) {
    if (allRows[i].style.display === 'none') continue;
    var cells = allRows[i].querySelectorAll('th, td');
    var rowVals = [];
    for (var j = 0; j < cells.length; j++) {
      var txt = cells[j].textContent.replace(/↕/g, '').trim().replace(/"/g, '""');
      rowVals.push('"' + txt + '"');
    }
    csvRows.push(rowVals.join(','));
  }
  var csvContent = '\uFEFF' + csvRows.join('\n');
  var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = (title || tableId) + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast('EXPORTED CSV: ' + (title || tableId));
}

function selectArchNode(modelId, nodeIdx) {
  APP_STATE.nodeStepState[modelId] = nodeIdx;
  var card = document.getElementById(modelId);
  if (!card) return;
  var items = card.querySelectorAll('.circuit-node');
  var activeName = '', activeNameEn = '', activeDesc = '', activeDescEn = '';
  var safeText = SiteSecurity.escapeHTML;
  for (var i = 0; i < items.length; i++) {
    var isTarget = (i === nodeIdx);
    items[i].classList.toggle('active', isTarget);
    if (isTarget) {
      activeName = items[i].getAttribute('data-node-name') || '';
      activeNameEn = items[i].getAttribute('data-node-name-en') || activeName;
      activeDesc = items[i].getAttribute('data-node-desc') || '';
      activeDescEn = items[i].getAttribute('data-node-desc-en') || '';
    }
  }
  var spotlight = document.getElementById('spotlight-' + modelId);
  if (spotlight && items.length > 0) {
    var sNum = (nodeIdx + 1) < 10 ? ('0' + (nodeIdx + 1)) : String(nodeIdx + 1);
    var tNum = items.length < 10 ? ('0' + items.length) : String(items.length);
    spotlight.innerHTML =
      '<span class="spotlight-badge">STAGE ' + sNum + ' / ' + tNum + ' · <span class="lang-zh">' + safeText(activeName) + '</span><span class="lang-en">' + safeText(activeNameEn) + '</span></span>' +
      '<span class="spotlight-text lang-zh">' + safeText(activeDesc) + '</span>' +
      '<span class="spotlight-text spotlight-text-en lang-en">' + safeText(activeDescEn) + '</span>';
  }
}

function stepArchNode(modelId, delta) {
  var card = document.getElementById(modelId);
  if (!card) return;
  var items = card.querySelectorAll('.circuit-node');
  if (!items.length) return;
  var cur = APP_STATE.nodeStepState[modelId] || 0;
  var next = (cur + delta + items.length) % items.length;
  selectArchNode(modelId, next);
}

function toggleAutoPlayNodes(modelId) {
  var btn = document.querySelector('[data-play-id="' + modelId + '"]');
  if (APP_STATE.autoPlayTimers[modelId]) {
    clearInterval(APP_STATE.autoPlayTimers[modelId]);
    delete APP_STATE.autoPlayTimers[modelId];
    if (btn) {
      btn.classList.remove('playing');
      btn.textContent = 'AUTO';
    }
    return;
  }
  if (btn) {
    btn.classList.add('playing');
    btn.textContent = 'STOP';
  }
  APP_STATE.autoPlayTimers[modelId] = setInterval(function() {
    stepArchNode(modelId, 1);
  }, 1500);
}

function setModelViewMode(mode) {
  APP_STATE.viewMode = mode;
  var compSec = document.getElementById('section-models-compendium');
  if (compSec) compSec.setAttribute('data-view-mode', mode);
  var btns = document.querySelectorAll('.view-mode-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.toggle('active', btns[i].getAttribute('data-view-btn') === mode);
  }
}

function setFamilyFilter(famId) {
  APP_STATE.familyFilter = famId;
  var pills = document.querySelectorAll('.fam-pill');
  for (var i = 0; i < pills.length; i++) {
    pills[i].classList.toggle('active', pills[i].getAttribute('data-fam-filter') === famId);
  }
  applyModelFilters();
}

function filterOnlyFamily(famId) {
  setFamilyFilter(famId);
  var sec = document.getElementById('section-models-compendium');
  if (sec) sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function onModelSearchInput(val) {
  APP_STATE.searchQuery = String(val || '').trim().toLowerCase();
  applyModelFilters();
}

function onEraFilterChange(val) {
  APP_STATE.eraFilter = val;
  applyModelFilters();
}

function onVenueFilterChange(val) {
  APP_STATE.venueFilter = val;
  applyModelFilters();
}

function onSpecialFilterChange(val) {
  APP_STATE.specialFilter = val;
  applyModelFilters();
}

function onSortOrderChange(val) {
  APP_STATE.sortOrder = val;
  sortModelCards(val);
}

function resetAllModelFilters() {
  APP_STATE.familyFilter = 'all';
  APP_STATE.eraFilter = 'all';
  APP_STATE.venueFilter = 'all';
  APP_STATE.specialFilter = 'all';
  APP_STATE.searchQuery = '';
  var sInput = document.getElementById('model-search-input');
  if (sInput) sInput.value = '';
  var eraSel = document.getElementById('era-filter-select');
  if (eraSel) eraSel.value = 'all';
  var venSel = document.getElementById('venue-filter-select');
  if (venSel) venSel.value = 'all';
  var spSel = document.getElementById('special-filter-select');
  if (spSel) spSel.value = 'all';
  setFamilyFilter('all');
}

function matchesModelFilter(el) {
  var mid = el.getAttribute('data-model-id');
  var fam = el.getAttribute('data-family-id');
  var era = el.getAttribute('data-era-group');
  var ven = el.getAttribute('data-venue-group');
  var hasCode = el.getAttribute('data-has-code') === '1';
  var searchBlob = (MODELS_MAP[mid] && MODELS_MAP[mid]._search) || '';

  if (APP_STATE.familyFilter !== 'all' && fam !== APP_STATE.familyFilter) return false;
  if (APP_STATE.eraFilter !== 'all' && era !== APP_STATE.eraFilter) return false;
  if (APP_STATE.venueFilter !== 'all' && ven !== APP_STATE.venueFilter) return false;
  if (APP_STATE.specialFilter === 'code' && !hasCode) return false;
  if (APP_STATE.specialFilter === 'bookmarked' && !APP_STATE.bookmarks[mid]) return false;
  if (APP_STATE.specialFilter === 'mastered' && !APP_STATE.mastered[mid]) return false;
  if (APP_STATE.searchQuery && searchBlob.indexOf(APP_STATE.searchQuery) === -1) return false;
  return true;
}

function applyModelFilters() {
  var cards = document.querySelectorAll('.model-card');
  var visibleCount = 0;
  var famCounts = {};

  for (var i = 0; i < cards.length; i++) {
    var c = cards[i];
    var show = matchesModelFilter(c);
    c.style.display = show ? '' : 'none';
    var fid = c.getAttribute('data-family-id');
    if (show) {
      visibleCount++;
      famCounts[fid] = (famCounts[fid] || 0) + 1;
    }
  }

  var mRows = document.querySelectorAll('.master-model-row');
  for (var r = 0; r < mRows.length; r++) {
    var row = mRows[r];
    var rMid = row.getAttribute('data-model-id');
    var correspondingCard = document.getElementById(rMid);
    var rShow = correspondingCard ? (correspondingCard.style.display !== 'none') : true;
    row.style.display = rShow ? '' : 'none';
  }

  var famSecs = document.querySelectorAll('.family-section');
  for (var j = 0; j < famSecs.length; j++) {
    var fs = famSecs[j];
    var fKey = fs.getAttribute('data-family-section');
    var cnt = famCounts[fKey] || 0;
    fs.style.display = cnt > 0 ? '' : 'none';
    var cntEl = fs.querySelector('.fam-visible-count');
    if (cntEl) cntEl.textContent = String(cnt);
  }

  var countLabel = document.getElementById('visible-models-count');
  if (countLabel) countLabel.textContent = String(visibleCount);

  var emptyState = document.getElementById('models-empty-state');
  if (emptyState) emptyState.style.display = visibleCount === 0 ? 'block' : 'none';
}

function sortModelCards(sortOrder) {
  var containers = document.querySelectorAll('.family-models-container');
  for (var i = 0; i < containers.length; i++) {
    var cont = containers[i];
    var cards = Array.prototype.slice.call(cont.querySelectorAll('.model-card'));
    cards.sort(function(a, b) {
      var numA = parseInt(a.getAttribute('data-model-num') || '0', 10);
      var numB = parseInt(b.getAttribute('data-model-num') || '0', 10);
      var yrA = parseInt(a.getAttribute('data-model-year') || '0', 10);
      var yrB = parseInt(b.getAttribute('data-model-year') || '0', 10);
      var tA = a.getAttribute('data-model-title') || '';
      var tB = b.getAttribute('data-model-title') || '';
      if (sortOrder === 'num-asc') return numA - numB;
      if (sortOrder === 'year-asc') return (yrA - yrB) || (numA - numB);
      if (sortOrder === 'year-desc') return (yrB - yrA) || (numA - numB);
      if (sortOrder === 'title-asc') return tA.localeCompare(tB, 'en');
      return numA - numB;
    });
    for (var k = 0; k < cards.length; k++) {
      cont.appendChild(cards[k]);
    }
  }
}

function toggleCollapseFamilySection(famId) {
  var cont = document.getElementById('container-' + famId);
  if (!cont) return;
  cont.style.display = (cont.style.display === 'none') ? '' : 'none';
}

function toggleTocFamily(famId) {
  var sub = document.getElementById('toc-sub-' + famId);
  if (!sub) return;
  sub.classList.toggle('open');
}

function filterTocItems(query) {
  var q = String(query || '').trim().toLowerCase();
  var famItems = document.querySelectorAll('.toc-family-item');
  for (var i = 0; i < famItems.length; i++) {
    var fi = famItems[i];
    var subList = fi.querySelector('.toc-submodel-list');
    var subLinks = fi.querySelectorAll('.toc-submodel-link');
    var anySubMatch = false;
    for (var j = 0; j < subLinks.length; j++) {
      var mMatch = !q || subLinks[j].textContent.toLowerCase().indexOf(q) !== -1;
      subLinks[j].parentNode.style.display = mMatch ? '' : 'none';
      if (mMatch) anySubMatch = true;
    }
    var famMatch = !q || fi.textContent.toLowerCase().indexOf(q) !== -1 || anySubMatch;
    fi.style.display = famMatch ? '' : 'none';
    if (subList) {
      if (q && anySubMatch) subList.classList.add('open');
      else if (!q) subList.classList.remove('open');
    }
  }
}

function filterTimelineYear(yr) {
  var btns = document.querySelectorAll('.yr-bar-item');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.toggle('active', btns[i].getAttribute('data-yr-filter') === String(yr));
  }
  var blocks = document.querySelectorAll('.tl-year-block');
  for (var j = 0; j < blocks.length; j++) {
    var bYr = blocks[j].getAttribute('data-tl-year');
    blocks[j].style.display = (yr === 'all' || String(yr) === bYr) ? '' : 'none';
  }
}

function syncPersonalStateUI() {
  var bmCount = Object.keys(APP_STATE.bookmarks).filter(function(k) { return APP_STATE.bookmarks[k]; }).length;
  var msCount = Object.keys(APP_STATE.mastered).filter(function(k) { return APP_STATE.mastered[k]; }).length;

  var bmBadge = document.getElementById('hero-bookmark-count');
  if (bmBadge) bmBadge.textContent = String(bmCount);
  var msBadge = document.getElementById('hero-mastered-count');
  if (msBadge) msBadge.textContent = String(msCount);

  var bmBtns = document.querySelectorAll('.bookmark-toggle-btn');
  for (var i = 0; i < bmBtns.length; i++) {
    var mid = bmBtns[i].getAttribute('data-bookmark-id');
    var active = !!APP_STATE.bookmarks[mid];
    bmBtns[i].classList.toggle('active', active);
  }

  var msBtns = document.querySelectorAll('.mastered-toggle-btn');
  for (var j = 0; j < msBtns.length; j++) {
    var mid2 = msBtns[j].getAttribute('data-mastered-id');
    var active2 = !!APP_STATE.mastered[mid2];
    msBtns[j].classList.toggle('active', active2);
  }
}

function toggleBookmarkModel(modelId) {
  APP_STATE.bookmarks[modelId] = !APP_STATE.bookmarks[modelId];
  if (!APP_STATE.bookmarks[modelId]) delete APP_STATE.bookmarks[modelId];
  safeSaveStorage();
  syncPersonalStateUI();
  var m = MODELS_MAP[modelId];
  showToast(APP_STATE.bookmarks[modelId] ? ('SAVED TO BOOKMARKS: ' + (m ? m.title : modelId)) : 'REMOVED FROM BOOKMARKS');
  if (APP_STATE.specialFilter === 'bookmarked') applyModelFilters();
}

function toggleMasteredModel(modelId) {
  APP_STATE.mastered[modelId] = !APP_STATE.mastered[modelId];
  if (!APP_STATE.mastered[modelId]) delete APP_STATE.mastered[modelId];
  safeSaveStorage();
  syncPersonalStateUI();
  var m = MODELS_MAP[modelId];
  showToast(APP_STATE.mastered[modelId] ? ('MARKED AS MASTERED: ' + (m ? m.title : modelId)) : 'UNMARKED MASTERED');
  if (APP_STATE.specialFilter === 'mastered') applyModelFilters();
}

function copyModelSummary(modelId) {
  var m = MODELS_MAP[modelId];
  if (!m) return;
  var lines = [
    '### [' + m.num_str + '/137] ' + m.title + ' (' + m.year_str + ' · ' + m.venue + ')',
    '- **架构族：** ' + m.family_label + ' (' + m.archetype + ')',
    '- **核心机制：** ' + m.core_mech,
    '- **输入 → 输出：** ' + m.input_desc + ' → ' + m.output_desc,
    '- **上一代瓶颈：** ' + (m.attrs['上一代瓶颈'] || ''),
    '- **核心想法：** ' + (m.attrs['核心想法'] || ''),
    '- **关键改进：** ' + (m.attrs['关键改进'] || ''),
    '- **医学适配：** ' + (m.attrs['医学适配'] || ''),
    '- **适用范围与证据：** ' + (m.attrs['适用范围与证据'] || ''),
    '- **原论文：** ' + m.paper_title + ' (' + m.paper_url + ')'
  ];
  copyTextSafe(lines.join('\n'), 'COPIED MARKDOWN SPECIFICATION: ' + m.title);
}

function openLightbox(modelId) {
  var m = MODELS_MAP[modelId];
  var card = document.getElementById(modelId);
  if (!m || !card) return;
  APP_STATE.lightboxModelId = modelId;
  APP_STATE.lightboxScale = 1;
  APP_STATE.lightboxTx = 0;
  APP_STATE.lightboxTy = 0;

  var imgEl = card.querySelector('.arch-figure-img');
  var lbImg = document.getElementById('lightbox-img');
  if (lbImg && imgEl) {
    lbImg.src = imgEl.src;
    lbImg.alt = imgEl.alt;
    updateLightboxTransform();
  }

  var titleEl = document.getElementById('lightbox-title');
  if (titleEl) {
    titleEl.textContent = 'SPEC ' + m.num_str + ' // ' + m.title + ' (' + m.year_str + ' · ' + m.venue + ') — ' + m.fig_badge;
  }

  var sideEl = document.getElementById('lightbox-side-content');
  if (sideEl) {
    var nodesHtml = m.nodes.map(function(n, idx) {
      var sIdx = (idx + 1) < 10 ? ('0' + (idx + 1)) : String(idx + 1);
      return '<div style="padding:8px 10px;background:var(--bg-elevated);border-bottom:1px solid var(--rule);font-size:12.5px">' +
             '<strong style="font-family:var(--font-mono);color:var(--accent-secondary)">' + sIdx + ' · ' + n.name + '：</strong>' + n.desc + '</div>';
    }).join('');
    sideEl.innerHTML =
      '<div style="padding:12px;border:1px solid var(--rule);background:var(--bg-elevated)"><div class="mono-kicker">CORE MECHANISM</div><p style="margin:4px 0 0;font-size:13px">' + m.core_mech + '</p></div>' +
      '<div class="mono-kicker">FORWARD COMPUTATION GRAPH (' + m.nodes.length + ' STAGES)</div>' +
      '<div style="border-top:1px solid var(--rule)">' + nodesHtml + '</div>';
  }

  var modal = document.getElementById('lightbox-modal');
  if (modal) modal.classList.add('open');
}

function closeLightbox() {
  var modal = document.getElementById('lightbox-modal');
  if (modal) modal.classList.remove('open');
}

function zoomLightbox(delta) {
  if (delta === 0) {
    APP_STATE.lightboxScale = 1;
    APP_STATE.lightboxTx = 0;
    APP_STATE.lightboxTy = 0;
  } else {
    APP_STATE.lightboxScale = Math.min(4.5, Math.max(0.4, APP_STATE.lightboxScale + delta));
  }
  updateLightboxTransform();
}

function updateLightboxTransform() {
  var lbImg = document.getElementById('lightbox-img');
  if (!lbImg) return;
  lbImg.style.transform = 'translate(' + APP_STATE.lightboxTx + 'px, ' + APP_STATE.lightboxTy + 'px) scale(' + APP_STATE.lightboxScale + ')';
  var zoomLbl = document.getElementById('lightbox-zoom-label');
  if (zoomLbl) zoomLbl.textContent = Math.round(APP_STATE.lightboxScale * 100) + '%';
}

function syncCompareUI() {
  var cnt = APP_STATE.compareList.length;
  var badge = document.getElementById('cmd-compare-count');
  if (badge) badge.textContent = String(cnt);
  var dockBadge = document.getElementById('dock-compare-count');
  if (dockBadge) dockBadge.textContent = String(cnt);

  var btns = document.querySelectorAll('.compare-toggle-btn');
  for (var i = 0; i < btns.length; i++) {
    var mid = btns[i].getAttribute('data-compare-id');
    var inList = APP_STATE.compareList.indexOf(mid) !== -1;
    btns[i].classList.toggle('active', inList);
  }
}

function toggleCompareModel(modelId) {
  var idx = APP_STATE.compareList.indexOf(modelId);
  var m = MODELS_MAP[modelId];
  if (idx !== -1) {
    APP_STATE.compareList.splice(idx, 1);
    showToast('REMOVED FROM COMPARISON: ' + (m ? m.title : modelId));
  } else {
    if (APP_STATE.compareList.length >= 4) {
      showToast('MAXIMUM 4 MODELS IN COMPARISON MATRIX');
      openCompareModal();
      return;
    }
    APP_STATE.compareList.push(modelId);
    showToast('ADDED TO COMPARISON (' + APP_STATE.compareList.length + '/4): ' + (m ? m.title : modelId));
  }
  syncCompareUI();
}

function loadPresetComparison(idList) {
  APP_STATE.compareList = idList.slice(0, 4);
  syncCompareUI();
  openCompareModal();
}

function clearCompareList() {
  APP_STATE.compareList = [];
  syncCompareUI();
  renderCompareModalBody();
}

function openCompareModal() {
  renderCompareModalBody();
  var modal = document.getElementById('compare-modal');
  if (modal) modal.classList.add('open');
}

function closeCompareModal() {
  var modal = document.getElementById('compare-modal');
  if (modal) modal.classList.remove('open');
}

function renderCompareModalBody() {
  var body = document.getElementById('compare-modal-body');
  if (!body) return;
  if (APP_STATE.compareList.length === 0) {
    body.innerHTML =
      '<div style="text-align:center;padding:44px 20px">' +
      '<div class="mono-kicker" style="margin-bottom:8px">COMPARISON MATRIX EMPTY</div>' +
      '<h4 style="margin:0 0 12px;font-family:var(--font-serif);font-size:22px">尚未选择待对比的模型</h4>' +
      '<p style="color:var(--ink-tertiary);margin:0 0 22px;font-size:13.5px">可在任意规格卡片顶部点击「COMPARE」，或直接加载以下典型跨范式对照组：</p>' +
      '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:8px">' +
      '<button type="button" class="btn-hairline" data-call-click="loadPresetComparison([\'architecture_model_002_resnet\',\'architecture_model_004_convnext\',\'architecture_model_021_vision_transformer__vit\',\'architecture_model_016_mamba\'])">01 // ResNet vs ConvNeXt vs ViT vs Mamba</button>' +
      '<button type="button" class="btn-hairline" data-call-click="loadPresetComparison([\'architecture_model_006_u_net\',\'architecture_model_009_nnu_net\',\'architecture_model_011_swin_unetr__ssl\',\'architecture_model_012_mednext\'])">02 // U-Net vs nnU-Net vs Swin UNETR vs MedNeXt</button>' +
      '<button type="button" class="btn-hairline" data-call-click="loadPresetComparison([\'architecture_model_050_modl\',\'architecture_model_051_varnet\',\'architecture_model_060_dps\',\'architecture_model_062_flow_matching\'])">03 // MoDL vs VarNet vs DPS vs Flow Matching</button>' +
      '</div></div>';
    return;
  }

  var selected = APP_STATE.compareList.map(function(id) { return MODELS_MAP[id]; }).filter(Boolean);
  var colHeaders = selected.map(function(m) {
    return '<td><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">' +
           '<div><span class="mono-kicker">SPEC ' + m.num_str + ' · ' + m.year_str + ' · ' + m.venue + '</span>' +
           '<div style="font-family:var(--font-serif);font-size:18px;font-weight:700;margin-top:2px">' + m.title + '</div></div>' +
           '<button type="button" class="btn-hairline" data-call-click="toggleCompareModel(\'' + m.id + '\'); renderCompareModalBody();">REMOVE</button>' +
           '</div></td>';
  }).join('');

  var figCells = selected.map(function(m) {
    var card = document.getElementById(m.id);
    var imgEl = card ? card.querySelector('.arch-figure-img') : null;
    var src = imgEl ? imgEl.src : '';
    return '<td><div style="background:#FFF;padding:8px;border:1px solid var(--rule);text-align:center">' +
           '<img src="' + src + '" alt="' + m.title + '" style="max-width:100%;max-height:170px;object-fit:contain;cursor:zoom-in" data-call-click="closeCompareModal(); openLightbox(\'' + m.id + '\')" />' +
           '</div></td>';
  }).join('');

  function makeRow(label, getter) {
    return '<tr><th>' + label + '</th>' + selected.map(function(m) {
      return '<td>' + (getter(m) || '—') + '</td>';
    }).join('') + '</tr>';
  }

  body.innerHTML =
    '<div class="table-responsive-wrap">' +
    '<table class="compare-matrix-table">' +
    '<tbody>' +
    '<tr><th>SPECIFICATION</th>' + colHeaders + '</tr>' +
    '<tr><th>ORIGINAL FIGURE</th>' + figCells + '</tr>' +
    makeRow('CORE MECHANISM // 核心机制', function(m) { return '<div class="lang-zh"><strong>' + m.core_mech + '</strong></div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:4px">' + (m.core_mech_en || '') + '</div>'; }) +
    makeRow('TENSOR FLOW // 输入输出', function(m) { return '<div class="lang-zh">' + m.input_desc + ' → ' + m.output_desc + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + (m.input_en || '') + ' → ' + (m.output_en || '') + '</div>'; }) +
    makeRow('PRIOR LIMITATION // 上一代瓶颈', function(m) { return '<div class="lang-zh">' + m.attrs['上一代瓶颈'] + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + ((m.attrs_en && m.attrs_en['上一代瓶颈']) || '') + '</div>'; }) +
    makeRow('CORE HYPOTHESIS // 核心想法', function(m) { return '<div class="lang-zh">' + m.attrs['核心想法'] + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + ((m.attrs_en && m.attrs_en['核心想法']) || '') + '</div>'; }) +
    makeRow('ARCHITECTURAL DELTA // 关键改进', function(m) { return '<div class="lang-zh">' + m.attrs['关键改进'] + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + ((m.attrs_en && m.attrs_en['关键改进']) || '') + '</div>'; }) +
    makeRow('MEDICAL ADAPTATION // 医学适配', function(m) { return '<div class="lang-zh">' + m.attrs['医学适配'] + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + ((m.attrs_en && m.attrs_en['医学适配']) || '') + '</div>'; }) +
    makeRow('EVIDENCE BOUNDARY // 证据边界', function(m) { return '<div class="lang-zh">' + m.attrs['适用范围与证据'] + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + ((m.attrs_en && m.attrs_en['适用范围与证据']) || '') + '</div>'; }) +
    makeRow('FORMALISM // 数学公式', function(m) { return '<div class="lang-zh">' + m.form_title + '</div><div class="lang-en" style="font-size:12px;color:var(--ink-secondary);margin-top:3px">' + (m.form_title_en || '') + '</div>'; }) +
    '</tbody></table></div>';
}

function expandSingleCardModal(modelId) {
  var m = MODELS_MAP[modelId];
  var card = document.getElementById(modelId);
  if (!m || !card) return;
  var fullBody = card.querySelector('.model-full-body');
  var modalBody = document.getElementById('quickview-modal-body');
  var modalTitle = document.getElementById('quickview-modal-title');
  if (modalTitle) {
    modalTitle.textContent = 'SPEC ' + m.num_str + ' / 137 // ' + m.title + ' (' + m.year_str + ' · ' + m.venue + ')';
  }
  if (modalBody && fullBody) {
    var clone = fullBody.cloneNode(true);
    var idEls = clone.querySelectorAll('[id]');
    for (var i = 0; i < idEls.length; i++) {
      idEls[i].removeAttribute('id');
    }
    modalBody.innerHTML = '';
    modalBody.appendChild(clone);
  }
  var modal = document.getElementById('quickview-modal');
  if (modal) modal.classList.add('open');
}

function closeQuickViewModal() {
  var modal = document.getElementById('quickview-modal');
  if (modal) modal.classList.remove('open');
}

function openSourceModal(sourceId) {
  var s = SOURCES_MAP[sourceId];
  var body = document.getElementById('source-modal-body');
  var title = document.getElementById('source-modal-title');
  if (!body) return;
  if (!s) {
    body.innerHTML = '<p>未找到指定数据源记录。</p>';
  } else {
    if (title) title.textContent = 'PROVENANCE // ' + (s.label || sourceId);
    var q = s.query || {};
    body.innerHTML =
      '<div style="display:grid;gap:12px">' +
      '<div><span class="mono-kicker">SOURCE ID: </span><code>' + s.id + '</code></div>' +
      '<div><span class="mono-kicker">ARTIFACT PATH: </span><code>' + (s.path || s.href || 'INTERNAL SNAPSHOT') + '</code></div>' +
      (q.description ? '<div><span class="mono-kicker">PROTOCOL NOTE: </span>' + q.description + '</div>' : '') +
      (q.sql ? '<div><span class="mono-kicker">SQL QUERY SNAPSHOT:</span><pre class="source-sql-box"><code>' + q.sql + '</code></pre></div>' : '') +
      '</div>';
  }
  var modal = document.getElementById('source-modal');
  if (modal) modal.classList.add('open');
}

function closeSourceModal() {
  var modal = document.getElementById('source-modal');
  if (modal) modal.classList.remove('open');
}

function openCitationModal() {
  var modal = document.getElementById('citation-modal');
  if (modal) modal.classList.add('open');
}

function closeCitationModal() {
  var modal = document.getElementById('citation-modal');
  if (modal) modal.classList.remove('open');
}

function openGlobalSearchModal() {
  var modal = document.getElementById('search-modal');
  if (modal) {
    modal.classList.add('open');
    var inp = document.getElementById('global-palette-input');
    if (inp) {
      inp.focus();
      onGlobalPaletteSearch(inp.value);
    }
  }
}

function closeGlobalSearchModal() {
  var modal = document.getElementById('search-modal');
  if (modal) modal.classList.remove('open');
}

function onGlobalPaletteSearch(query) {
  var q = String(query || '').trim().toLowerCase();
  var resBox = document.getElementById('global-palette-results');
  if (!resBox) return;
  var matches = [];
  for (var i = 0; i < window.MODELS_META.length; i++) {
    var m = window.MODELS_META[i];
    var hay = (m.title + ' ' + m.num_str + ' ' + m.year_str + ' ' + m.venue + ' ' + m.family_label + ' ' + (m.family_en || '') + ' ' + m.core_mech + ' ' + (m.core_mech_en || '') + ' ' + (m.attrs['医学适配'] || '') + ' ' + ((m.attrs_en && m.attrs_en['医学适配']) || '')).toLowerCase();
    if (!q || hay.indexOf(q) !== -1) {
      matches.push(m);
      if (matches.length >= 24) break;
    }
  }
  if (matches.length === 0) {
    resBox.innerHTML = '<div style="padding:24px;text-align:center;color:var(--ink-tertiary);font-family:var(--font-mono);font-size:12px">NO MATCHING ARCHITECTURE FOUND</div>';
    return;
  }
  resBox.innerHTML = matches.map(function(m) {
    return '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 18px;border-bottom:1px solid var(--rule);cursor:pointer" data-call-click="closeGlobalSearchModal(); jumpToModelCard(\'' + m.id + '\')">' +
           '<div><span class="mono-kicker">SPEC ' + m.num_str + ' · ' + m.year_str + ' · ' + m.venue + ' · ' + m.family_short + '</span>' +
           '<div style="font-family:var(--font-serif);font-size:17px;font-weight:700;margin-top:2px">' + m.title + '</div>' +
           '<div class="lang-zh" style="font-size:12.5px;color:var(--ink-secondary);margin-top:2px">' + m.core_mech + '</div>' +
           '<div class="lang-en" style="font-size:12px;color:var(--ink-tertiary);margin-top:2px">' + (m.core_mech_en || '') + '</div></div>' +
           '<span class="mono-kicker" style="color:var(--accent-primary)">INSPECT ↵</span></div>';
  }).join('');
}

function initScrollAndEvents() {
  var progressPending = false;
  window.addEventListener('scroll', function() {
    if (progressPending) return; progressPending = true;
    requestAnimationFrame(function() {
    progressPending = false;
    var docEl = document.documentElement;
    var scrollTop = window.scrollY || docEl.scrollTop;
    var scrollHeight = docEl.scrollHeight - docEl.clientHeight;
    var pct = scrollHeight > 0 ? Math.min(100, (scrollTop / scrollHeight) * 100) : 0;
    var bar = document.getElementById('reading-progress-bar');
    if (bar) bar.style.width = pct + '%';
    });
  }, { passive: true });

  var stage = document.getElementById('lightbox-stage');
  if (stage) {
    stage.addEventListener('wheel', function(e) {
      e.preventDefault();
      zoomLightbox(e.deltaY < 0 ? 0.2 : -0.2);
    });
    stage.addEventListener('mousedown', function(e) {
      APP_STATE.isDraggingLb = true;
      APP_STATE.dragStartX = e.clientX - APP_STATE.lightboxTx;
      APP_STATE.dragStartY = e.clientY - APP_STATE.lightboxTy;
    });
    window.addEventListener('mousemove', function(e) {
      if (!APP_STATE.isDraggingLb) return;
      APP_STATE.lightboxTx = e.clientX - APP_STATE.dragStartX;
      APP_STATE.lightboxTy = e.clientY - APP_STATE.dragStartY;
      updateLightboxTransform();
    });
    window.addEventListener('mouseup', function() {
      APP_STATE.isDraggingLb = false;
    });
  }

  window.addEventListener('keydown', function(e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      openGlobalSearchModal();
    } else if (e.key === 'Escape') {
      closeLightbox();
      closeCompareModal();
      closeQuickViewModal();
      closeSourceModal();
      closeCitationModal();
      closeGlobalSearchModal();
    }
  });
}

window.addEventListener('DOMContentLoaded', function() {
  if (window.MODELS_META) {
    for (var i = 0; i < window.MODELS_META.length; i++) {
      var m = window.MODELS_META[i];
      m._search = flattenSearchValues(m).join(' ').toLowerCase();
      MODELS_MAP[m.id] = m;
    }
  }
  if (window.SOURCES_META) {
    for (var j = 0; j < window.SOURCES_META.length; j++) {
      var s = window.SOURCES_META[j];
      SOURCES_MAP[s.id] = s;
    }
  }
  if (window.WORKFLOWS_META) {
    WORKFLOWS_DATA = window.WORKFLOWS_META;
  }
  safeLoadStorage();
  setTheme(APP_STATE.theme);
  if (APP_STATE.font === 'academic') {
    document.documentElement.setAttribute('data-font', 'academic');
  }
  var allCards = document.querySelectorAll('.model-card');
  for (var cIdx = 0; cIdx < allCards.length; cIdx++) {
    var mainIm = allCards[cIdx].querySelector('.arch-figure-img');
    var thumbIm = allCards[cIdx].querySelector('.gallery-thumb-img');
    if (mainIm && thumbIm) thumbIm.src = mainIm.src;
  }
  syncPersonalStateUI();
  syncCompareUI();
  selectPipelineStage(0);
  selectWizardWorkflow(0);
  initScrollAndEvents();
});

function setLanguageMode(mode) {
  var valid = (mode === 'zh' || mode === 'en' || mode === 'bilingual') ? mode : 'bilingual';
  APP_STATE.lang = valid;
  document.documentElement.setAttribute('data-lang', valid);
  var btns = document.querySelectorAll('.lang-seg-btn');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.toggle('active', btns[i].getAttribute('data-lang-btn') === valid);
  }
  try {
    localStorage.setItem('biomed_atlas_lang_v3', valid);
  } catch (e) {}
  var labels = {
    'bilingual': 'LANGUAGE MODE: 中英对照 BILINGUAL',
    'zh': 'LANGUAGE MODE: 中文原著 ZH',
    'en': 'LANGUAGE MODE: ENGLISH EDITION EN'
  };
  showToast(labels[valid] || 'LANGUAGE UPDATED');
}

function openDeployModal() {
  var modal = document.getElementById('deploy-modal');
  if (modal) modal.classList.add('open');
}

function closeDeployModal() {
  var modal = document.getElementById('deploy-modal');
  if (modal) modal.classList.remove('open');
}

function copyDeployCode(preId) {
  var el = document.getElementById(preId);
  if (!el) return;
  copyTextSafe(el.textContent || el.innerText, 'COPIED GITHUB DEPLOYMENT COMMANDS');
}


