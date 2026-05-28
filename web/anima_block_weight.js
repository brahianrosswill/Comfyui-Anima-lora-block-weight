// Anima LoRA Block Weight V2 — 单面板统一 UI 前端
// =====================================================================
// 关键架构（根治布局打架）：
//   节点里只保留两个原生 widget：lora_name、control_mode。
//   其它所有控件（strength×2、四段区间×4、四段权重×4、w_×4、verbose、blk00-27）
//   全部隐藏其原生 widget，统一塞进【一个】DOM 面板，由该面板内部用 CSS flex 自行布局。
//   这样 ComfyUI 只需给这一个 DOM 面板分配总高度，不再有"原生 widget 与 DOM widget
//   垂直交错、各算各高度"导致的挤压/留白问题。
//
// 面板内部结构（grouped）：
//   strength_model / strength_clip            （名称 + [◄][数值][►]）
//   ── 分隔 ──
//   seg_motion   : [区间文本] + 方块+滑块+数值 + impact染色
//   seg_proportion / seg_core / seg_detail     （同上）
//   ── 分隔 ──
//   w_self_attn / w_cross_attn / w_mlp / w_adaln（方块+滑块+数值，不染色）
//   ── 分隔 ──
//   verbose（开关）
// per_block 模式：隐藏四段区间/权重行，显示 28 个 blk 行（带 impact 染色）+ 工具条。
//
// 原生 widget 始终作为数据底座（JS 写回其 .value）；JS 失效时原生控件仍可用。
// i18n 按 Comfy.Locale 自动中/英。兼容 ComfyUI 0.20.1 / 前端 1.42.15。

import { app } from "../../scripts/app.js";

const NODE_NAMES = ["AnimaLoRABlockWeightV2", "AnimaLoRABlockWeightExport"];
const TOTAL = 28;
const NODE_W = 400;
const ROW_H = 20;

// 注入一次全局样式：关闭 number input 的原生上下微调箭头（用我们自己的 ◄ ► 代替）
(function injectStyle() {
  if (document.getElementById("anima-lbw-style")) return;
  const s = document.createElement("style");
  s.id = "anima-lbw-style";
  s.textContent =
    ".anima-no-spin::-webkit-outer-spin-button,.anima-no-spin::-webkit-inner-spin-button{-webkit-appearance:none;margin:0;}" +
    ".anima-no-spin{-moz-appearance:textfield;}";
  document.head.appendChild(s);
})();

// ---------- i18n ----------
function isZh() {
  try {
    const loc = app?.ui?.settings?.getSettingValue?.("Comfy.Locale");
    if (typeof loc === "string") return loc.toLowerCase().startsWith("zh");
  } catch (e) {}
  try { return (navigator.language || "").toLowerCase().startsWith("zh"); } catch (e) {}
  return false;
}
const I18N = {
  allOn:   { zh: "全开=1.0", en: "All On=1.0" },
  allOff:  { zh: "全关=0",   en: "All Off=0" },
  refresh: { zh: "刷新",     en: "Refresh" },
  impact:  { zh: "影响力",   en: "impact" },
  verbose: { zh: "verbose (控制台打印)", en: "verbose (console log)" },
};
function t(k) { const e = I18N[k]; return e ? (isZh() ? e.zh : e.en) : k; }

// ---------- 颜色 ----------
function impactColor(v) {
  if (v == null || isNaN(v)) return "#666";
  v = Math.max(0, Math.min(1, v));
  let r, g, b;
  if (v < 0.33) { const tt = v / 0.33; r = 40; g = Math.round(120 + 135 * tt); b = 230; }
  else if (v < 0.66) { const tt = (v - 0.33) / 0.33; r = Math.round(40 + 215 * tt); g = 255; b = Math.round(230 - 230 * tt); }
  else { const tt = (v - 0.66) / 0.34; r = 255; g = Math.round(255 - 195 * tt); b = 0; }
  return `rgb(${r},${g},${b})`;
}
function impactBg(v) { if (v == null || isNaN(v)) return "transparent"; return impactColor(v).replace("rgb(", "rgba(").replace(")", ",0.22)"); }

// ---------- 区间解析 ----------
function parseRange(str) {
  const out = [];
  (str || "").split(",").forEach((part) => {
    part = part.trim(); if (!part) return;
    if (part.includes("-")) { const [a, b] = part.split("-").map((x) => parseInt(x)); if (!isNaN(a) && !isNaN(b)) for (let i = Math.min(a, b); i <= Math.max(a, b); i++) out.push(i); }
    else { const i = parseInt(part); if (!isNaN(i)) out.push(i); }
  });
  return out;
}

// ---------- widget 工具 ----------
function findW(node, name) { return node.widgets?.find((x) => x.name === name) || null; }
function hideWidget(w) {
  if (!w) return;
  if (w.type && !w._origType) w._origType = w.type;
  w.hidden = true; w.type = "hidden" + (w._origType ? ":" + w._origType : "");
  w.computeSize = () => [0, -4];
}
function setW(w, val) { if (w) { w.value = val; if (w.callback) { try { w.callback(val); } catch (e) {} } } }

// ---------- [◄][num][►] 步进数值框 ----------
function makeNumStepper(getVal, setVal, { min = 0, max = 2, step = 0.05 } = {}) {
  const box = document.createElement("div");
  box.style.cssText = "display:flex;align-items:center;gap:2px;flex:0 0 auto;";
  const sync = () => { num.value = Number(getVal()).toFixed(2); };
  const mkArrow = (txt, delta) => {
    const b = document.createElement("button");
    b.textContent = txt;
    b.style.cssText = "width:16px;height:17px;line-height:15px;text-align:center;background:#3a3a3a;color:#ccc;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:10px;padding:0;flex:0 0 auto;";
    b.addEventListener("click", (e) => { e.preventDefault(); let v = getVal() + delta; v = Math.max(min, Math.min(max, Math.round(v * 100) / 100)); setVal(v); sync(); });
    return b;
  };
  const left = mkArrow("\u25C4", -step);
  const num = document.createElement("input");
  num.type = "number"; num.min = min; num.max = max; num.step = step;
  num.style.cssText = "flex:0 0 44px;background:#222;color:#fff;border:1px solid #444;border-radius:3px;text-align:center;font-family:monospace;padding:1px 2px;font-size:10px;-moz-appearance:textfield;appearance:textfield;";
  // 关掉原生上下微调箭头（webkit）
  num.classList.add("anima-no-spin");
  const right = mkArrow("\u25BA", step);
  num.addEventListener("change", () => { let v = parseFloat(num.value); if (isNaN(v)) v = getVal(); v = Math.max(min, Math.min(max, v)); setVal(v); sync(); });
  box.appendChild(left); box.appendChild(num); box.appendChild(right);
  sync();
  return { box, sync };
}

function rowBase() {
  const row = document.createElement("div");
  row.style.cssText = `display:flex;align-items:center;gap:5px;min-height:${ROW_H}px;border-radius:3px;padding:1px 3px;box-sizing:border-box;`;
  return row;
}

// ---------- strength 行 ----------
function makeStrengthRow(node, widgetName) {
  const w = findW(node, widgetName);
  const row = rowBase();
  const label = document.createElement("span");
  label.textContent = widgetName;
  label.style.cssText = "flex:1 1 auto;color:#ddd;font-family:monospace;font-size:11px;user-select:none;";
  const stepper = makeNumStepper(() => (w ? w.value : 1.0), (v) => setW(w, v), { min: -10, max: 10, step: 0.05 });
  row.appendChild(label); row.appendChild(stepper.box);
  return { row, sync: stepper.sync };
}

// ---------- 权重行（方块+名称+滑块+数值），可选染色 ----------
function makeWeightRow(node, widgetName, labelText, { colorable = false, labelW = 90 } = {}) {
  const w = findW(node, widgetName);
  const row = rowBase();
  const chk = document.createElement("input");
  chk.type = "checkbox";
  chk.style.cssText = "cursor:pointer;margin:0;flex:0 0 auto;width:13px;height:13px;accent-color:#888;";
  chk.checked = w ? w.value > 0 : true;
  const label = document.createElement("span");
  label.textContent = labelText;
  label.style.cssText = `flex:0 0 ${labelW}px;color:#ddd;font-family:monospace;font-size:10px;user-select:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`;
  const slider = document.createElement("input");
  slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.01";
  slider.value = w ? w.value : 1.0;
  slider.style.cssText = "flex:1 1 auto;cursor:pointer;min-width:30px;";
  let lastOn = w && w.value > 0 ? w.value : 1.0;
  const writeBack = (v) => setW(w, v);
  const stepper = makeNumStepper(() => (w ? w.value : 1.0),
    (v) => { writeBack(v); slider.value = v; if (v > 0) { lastOn = v; chk.checked = true; } else chk.checked = false; setEnabled(); },
    { min: 0, max: 2, step: 0.05 });
  const setEnabled = () => { const on = chk.checked; slider.disabled = !on; const op = on ? "1" : "0.4"; label.style.opacity = op; slider.style.opacity = op; stepper.box.style.opacity = op; };
  chk.addEventListener("change", () => {
    if (chk.checked) { const v = lastOn > 0 ? lastOn : 1.0; slider.value = v; writeBack(v); stepper.sync(); }
    else { if ((w ? w.value : 0) > 0) lastOn = w.value; slider.value = 0; writeBack(0); stepper.sync(); }
    setEnabled();
  });
  slider.addEventListener("input", () => { const v = parseFloat(slider.value); if (v > 0) lastOn = v; writeBack(v); stepper.sync(); chk.checked = v > 0; setEnabled(); });
  row.appendChild(chk); row.appendChild(label); row.appendChild(slider); row.appendChild(stepper.box);
  setEnabled();
  const refresh = () => { const v = w ? w.value : 1.0; slider.value = v; stepper.sync(); chk.checked = v > 0; if (v > 0) lastOn = v; setEnabled(); };
  const setColor = (imp) => { if (!colorable) return; row.style.background = impactBg(imp); chk.style.accentColor = (imp == null || isNaN(imp)) ? "#888" : impactColor(imp); row.title = imp != null ? `${t("impact")} ${(imp * 100).toFixed(0)}%` : ""; };
  return { row, refresh, setColor, widgetName };
}

// ---------- 区间文本行（seg_*_blocks） ----------
function makeRangeRow(node, widgetName) {
  const w = findW(node, widgetName);
  const row = rowBase();
  const label = document.createElement("span");
  label.textContent = widgetName;
  label.style.cssText = "flex:1 1 auto;color:#aaa;font-family:monospace;font-size:10px;user-select:none;";
  const inp = document.createElement("input");
  inp.type = "text";
  inp.value = w ? w.value : "";
  inp.style.cssText = "flex:0 0 78px;background:#222;color:#fff;border:1px solid #444;border-radius:3px;text-align:center;font-family:monospace;padding:1px 3px;font-size:10px;";
  inp.addEventListener("change", () => { setW(w, inp.value); if (node._applyImpactAgain) node._applyImpactAgain(); });
  row.appendChild(label); row.appendChild(inp);
  const refresh = () => { inp.value = w ? w.value : ""; };
  return { row, refresh };
}

// ---------- verbose 开关行 ----------
function makeToggleRow(node, widgetName, labelText) {
  const w = findW(node, widgetName);
  const row = rowBase();
  const chk = document.createElement("input");
  chk.type = "checkbox";
  chk.style.cssText = "cursor:pointer;margin:0;flex:0 0 auto;width:13px;height:13px;accent-color:#888;";
  chk.checked = !!(w && w.value);
  const label = document.createElement("span");
  label.textContent = labelText;
  label.style.cssText = "flex:1 1 auto;color:#ddd;font-family:monospace;font-size:11px;user-select:none;";
  chk.addEventListener("change", () => setW(w, chk.checked));
  row.appendChild(chk); row.appendChild(label);
  const refresh = () => { chk.checked = !!(w && w.value); };
  return { row, refresh };
}

function sep() { const d = document.createElement("div"); d.style.cssText = "height:1px;background:#444;margin:3px 0;flex:0 0 auto;"; return d; }
function mkBtn(txt, fn) {
  const b = document.createElement("button");
  b.textContent = txt;
  b.style.cssText = "flex:1 1 0;background:#333;color:#ddd;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:10px;padding:2px 4px;white-space:nowrap;";
  b.addEventListener("click", (e) => { e.preventDefault(); fn(); });
  return b;
}

// ---------- 构建唯一的统一面板 ----------
function buildPanel(node) {
  if (node._panel) return;
  const wrap = document.createElement("div");
  wrap.style.cssText = "width:100%;display:flex;flex-direction:column;gap:2px;box-sizing:border-box;";

  // strength（仅加载节点有；导出节点没有则跳过）
  node._strRows = [];
  ["strength_model", "strength_clip"].forEach((nm) => {
    if (!findW(node, nm)) return;
    const r = makeStrengthRow(node, nm); wrap.appendChild(r.row); node._strRows.push(r);
  });
  if (node._strRows.length) wrap.appendChild(sep());

  // 四段：每段 = 区间文本行 + 权重行
  node._segRangeRows = [];
  node._segRows = [];
  const segDefs = [
    ["seg_motion_blocks", "seg_motion_weight", "seg_motion"],
    ["seg_proportion_blocks", "seg_proportion_weight", "seg_proportion"],
    ["seg_core_blocks", "seg_core_weight", "seg_core"],
    ["seg_detail_blocks", "seg_detail_weight", "seg_detail"],
  ];
  node._segGroup = document.createElement("div");
  node._segGroup.style.cssText = "display:flex;flex-direction:column;gap:1px;";
  segDefs.forEach(([blocksName, weightName, label]) => {
    const rr = makeRangeRow(node, blocksName);
    const wr = makeWeightRow(node, weightName, label, { colorable: true, labelW: 90 });
    wr.blocksName = blocksName;
    node._segGroup.appendChild(rr.row);
    node._segGroup.appendChild(wr.row);
    node._segRangeRows.push(rr);
    node._segRows.push(wr);
  });
  wrap.appendChild(node._segGroup);

  node._segSep = sep();
  wrap.appendChild(node._segSep);

  // w_ 系数
  node._wRows = [];
  ["w_self_attn", "w_cross_attn", "w_mlp", "w_adaln"].forEach((nm) => { const r = makeWeightRow(node, nm, nm, { colorable: false, labelW: 90 }); wrap.appendChild(r.row); node._wRows.push(r); });

  wrap.appendChild(sep());

  // verbose（仅加载节点有）
  node._verboseRow = null;
  if (findW(node, "verbose")) {
    node._verboseRow = makeToggleRow(node, "verbose", t("verbose"));
    wrap.appendChild(node._verboseRow.row);
  }

  // per_block 区（工具条 + 28 行，默认隐藏）
  node._pbGroup = document.createElement("div");
  node._pbGroup.style.cssText = "display:none;flex-direction:column;gap:2px;";
  const pbSep = sep(); node._pbGroup.appendChild(pbSep);
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;gap:4px;flex:0 0 auto;";
  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:1px;overflow-y:auto;overflow-x:hidden;width:100%;box-sizing:border-box;";
  node._pbRows = [];
  for (let i = 0; i < TOTAL; i++) {
    const r = makeWeightRow(node, `blk${String(i).padStart(2, "0")}`, "blk" + String(i).padStart(2, "0"), { colorable: true, labelW: 38 });
    list.appendChild(r.row); node._pbRows.push(r);
  }
  const setAll = (val) => { node._pbRows.forEach((r) => setW(findW(node, r.widgetName), val)); node._pbRows.forEach((r) => r.refresh()); };
  toolbar.appendChild(mkBtn(t("allOn"), () => setAll(1.0)));
  toolbar.appendChild(mkBtn(t("allOff"), () => setAll(0.0)));
  toolbar.appendChild(mkBtn(t("refresh"), () => node._pbRows.forEach((r) => r.refresh())));
  node._pbGroup.appendChild(toolbar);
  node._pbGroup.appendChild(list);
  wrap.appendChild(node._pbGroup);
  node._pbList = list;

  node._panel = node.addDOMWidget("__anima_panel", "div", wrap, { serialize: false, hideOnZoom: false, getValue() { return ""; }, setValue() {} });
  node._panelWrap = wrap;

  // 让 ComfyUI 按面板实际内容高度分配（唯一的 DOM widget，不再有混合堆叠）
  node._panel.computeSize = function (width) {
    const modeW = findW(node, "control_mode");
    const isPB = modeW && modeW.value === "per_block";
    let rows = 0;
    rows += node._strRows ? node._strRows.length : 0;  // strength（加载节点 2，导出节点 0）
    if (!isPB) rows += 8;                               // 四段：区间4 + 权重4
    rows += 4;                                          // w_ ×4
    if (node._verboseRow) rows += 1;                    // verbose（仅加载节点）
    let h = rows * (ROW_H + 2) + 4 * 7;
    if (isPB) {
      const pbMinList = 120; // 列表最小高度（约 6 行），可继续拖矮
      h += 26 + pbMinList + 8;
    }
    return [width, h + 8];
  };
}

function segImpact(node, blocksName, impact) {
  const w = findW(node, blocksName);
  const idxs = parseRange(w ? w.value : "");
  const vals = idxs.map((i) => impact[i]).filter((v) => v != null && !isNaN(v));
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
function applyImpact(node, impact) {
  if (!Array.isArray(impact)) return;
  node._impact = impact;
  if (node._pbRows) node._pbRows.forEach((r, i) => r.setColor(impact[i]));
  if (node._segRows) node._segRows.forEach((r) => r.setColor(segImpact(node, r.blocksName, impact)));
}
function refreshAll(node) {
  node._strRows?.forEach((r) => r.sync());
  node._segRangeRows?.forEach((r) => r.refresh());
  node._segRows?.forEach((r) => r.refresh());
  node._wRows?.forEach((r) => r.refresh());
  node._verboseRow?.refresh();
  node._pbRows?.forEach((r) => r.refresh());
}

function toggleMode(node) {
  // 隐藏所有被接管的原生 widget（只留 lora_name、control_mode）
  ["strength_model", "strength_clip", "w_self_attn", "w_cross_attn", "w_mlp", "w_adaln",
   "seg_motion_weight", "seg_proportion_weight", "seg_core_weight", "seg_detail_weight",
   "seg_motion_blocks", "seg_proportion_blocks", "seg_core_blocks", "seg_detail_blocks",
   "verbose"].forEach((n) => hideWidget(findW(node, n)));
  for (let i = 0; i < TOTAL; i++) hideWidget(findW(node, `blk${String(i).padStart(2, "0")}`));

  if (!node._panel) buildPanel(node);

  const modeW = findW(node, "control_mode");
  const isPB = modeW && modeW.value === "per_block";

  // 段组与 per_block 组的显隐
  if (node._segGroup) node._segGroup.style.display = isPB ? "none" : "flex";
  if (node._segSep) node._segSep.style.display = isPB ? "none" : "block";
  if (node._pbGroup) node._pbGroup.style.display = isPB ? "flex" : "none";

  node._applyImpactAgain = () => { if (node._impact) applyImpact(node, node._impact); };

  refreshAll(node);
  if (node._impact) applyImpact(node, node._impact);

  // computeSize 返回的是【最小】高度；setSize 设【当前】高度。
  // grouped：当前=最小（精确贴合内容）。
  // per_block：当前给一个舒适高度（能看到较多行），但最小很小（可拖矮、列表内部滚动）。
  const cs = node.computeSize ? node.computeSize() : null;
  if (cs) {
    if (isPB) {
      const comfortable = cs[1] + (TOTAL - 6) * (ROW_H + 1);
      node.setSize([Math.max(node.size[0], NODE_W), Math.min(comfortable, 900)]);
    } else {
      node.setSize([Math.max(node.size[0], NODE_W), cs[1]]);
    }
  }
  if (isPB) syncPanelHeight(node);
  node.setDirtyCanvas(true, true);
}

function syncPanelHeight(node) {
  if (!node._pbList || !node._pbGroup) return;
  // per_block 列表 = 节点高度 - 列表顶部在节点内的位置 - 底部余量
  const nodeH = node.size ? node.size[1] : 600;
  let listTop = 300;
  try {
    // _pbList 相对节点顶部的位置：用面板 last_y + 列表在面板内的 offsetTop
    if (node._panel && node._panel.last_y != null) {
      listTop = node._panel.last_y + (node._pbList.offsetTop || 0);
    }
  } catch (e) {}
  const avail = Math.max(60, nodeH - listTop - 24);
  node._pbList.style.maxHeight = avail + "px";
}

app.registerExtension({
  name: "anima.lora.block.weight.v2.singlepanel",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (!NODE_NAMES.includes(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const self = this;
      const modeW = findW(this, "control_mode");
      if (modeW) {
        const orig = modeW.callback;
        modeW.callback = function () { const ret = orig ? orig.apply(this, arguments) : undefined; toggleMode(self); return ret; };
      }
      setTimeout(() => toggleMode(self), 0);
      return r;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) { const r = onResize ? onResize.apply(this, arguments) : undefined; syncPanelHeight(this); return r; };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
      try { const modeW = findW(this, "control_mode"); if (modeW && modeW.value === "per_block" && this._panel) syncPanelHeight(this); } catch (e) {}
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () { const r = onConfigure ? onConfigure.apply(this, arguments) : undefined; const self = this; setTimeout(() => toggleMode(self), 0); return r; };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      try { let imp = message?.block_impact; if (Array.isArray(imp) && imp.length && Array.isArray(imp[0])) imp = imp[0]; if (Array.isArray(imp) && imp.length) applyImpact(this, imp); } catch (e) {}
      return r;
    };
  },
});
