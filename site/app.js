/* by1 架构浏览器 -- 纯静态，无构建。
 *
 * 数据：data/index.js 定义 window.BY1_INDEX；
 *       每份模型 data/<slug>.js 定义 window.BY1_MODELS[slug]。
 * 为什么不是 fetch(json)：file:// 下 fetch 被 CORS 拦死，script 标签不会。
 */
(function () {
  "use strict";

  var INDEX = window.BY1_INDEX || [];
  var MODELS = window.BY1_MODELS || {};
  var current = null;
  var filter = "";

  function $(id) { return document.getElementById(id); }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = function () { res(); };
      s.onerror = function () { rej(new Error("加载失败: " + src)); };
      document.head.appendChild(s);
    });
  }

  function kindClass(kind) {
    var ok = { Norm: 1, Attention: 1, FFN: 1, MoE: 1, Linear: 1, SSM: 1, MLA: 1 };
    return ok[kind] ? "k-" + kind : "k-other";
  }

  function shortAttrs(op) {
    var a = op.attrs || {};
    var parts = [];
    if (a.q != null && a.kv != null) parts.push(a.q + "q/" + a.kv + "kv");
    if (a.head_dim != null) parts.push("hd" + a.head_dim);
    if (a.experts != null) parts.push(a.experts + "e/top" + (a.top_k != null ? a.top_k : "?"));
    if (a.hidden != null) parts.push("h" + a.hidden);
    if (a.window != null) parts.push("win" + a.window);
    if (a.one_plus) parts.push("1+w");
    return parts.join(" · ");
  }

  function renderList() {
    var ul = $("models");
    ul.innerHTML = "";
    var f = filter.toLowerCase();
    var shown = 0;
    INDEX.forEach(function (m) {
      if (f && (m.name + " " + (m.arch || "")).toLowerCase().indexOf(f) < 0) return;
      shown += 1;
      var li = document.createElement("li");
      if (m.error) {
        li.innerHTML = '<span class="nm">' + esc(m.name) + '</span>' +
          '<span class="meta bad">导不出：' + esc(m.error) + '</span>';
      } else {
        li.innerHTML = '<span class="nm">' + esc(m.name) + '</span>' +
          '<span class="meta">' + esc(m.arch || "-") + " · " + m.layers + " 层" +
          (m.n_tensors ? " · " + m.n_tensors + " 张量" : "") + "</span>";
        li.onclick = function () { select(m.slug); };
        if (current === m.slug) li.className = "active";
      }
      ul.appendChild(li);
    });
    $("count").textContent = shown + " / " + INDEX.length + " 份";
  }

  function select(slug) {
    if (!window.BY1_MODELS || !window.BY1_MODELS[slug]) {
      $("model").hidden = true;
      $("empty").hidden = false;
      $("empty").innerHTML = "<p>正在加载 " + esc(slug) + " …</p>";
      loadScript("data/" + slug + ".js").then(function () {
        MODELS = window.BY1_MODELS;
        select(slug);
      }).catch(function (e) {
        $("empty").innerHTML = "<p class='bad'>" + esc(e.message) + "</p>";
      });
      return;
    }
    current = slug;
    renderList();
    var m = window.BY1_MODELS[slug];
    $("empty").hidden = true;
    $("model").hidden = false;
    renderHead(m);
    renderLayers(m);
    $("props").hidden = true;
    $("props-empty").hidden = false;
  }

  function renderHead(m) {
    var h = $("mhead");
    var bits = [];
    bits.push("<span>arch <b>" + esc(m.arch || "-") + "</b></span>");
    if (m.ctx) bits.push("<span>ctx <b>" + esc(m.ctx) + "</b></span>");
    if (m.d_model) bits.push("<span>d_model <b>" + esc(m.d_model) + "</b></span>");
    if (m.vocab) bits.push("<span>vocab <b>" + esc(m.vocab) + "</b></span>");
    bits.push("<span>归一 <b>" + esc(m.norm_kind || "-") +
      (m.norm_one_plus ? " (1+w)" : " (w)") + "</b></span>");
    if (m.n_tensors) bits.push("<span>声明张量条目 <b>" + esc(m.n_tensors) + "</b></span>");
    var hp = m.hparams || {};
    var hpk = Object.keys(hp).map(function (k) {
      return esc(k) + "=" + esc(hp[k]);
    }).join(" · ");
    h.innerHTML = "<h2>" + esc(m.name) + "</h2><div class='kv'>" + bits.join("") + "</div>" +
      (hpk ? "<div class='kv' style='margin-top:6px'>hparams: " + hpk + "</div>" : "");
  }

  function renderLayers(m) {
    var box = $("layers");
    box.innerHTML = "";
    (m.layers || []).forEach(function (L) {
      var row = document.createElement("div");
      row.className = "layer";
      var no = document.createElement("div");
      no.className = "lno";
      no.textContent = "L" + L.index;
      var chain = document.createElement("div");
      chain.className = "chain";
      (L.ops || []).forEach(function (op, i) {
        if (i) {
          var ar = document.createElement("span");
          ar.className = "arrow";
          ar.textContent = "→";
          chain.appendChild(ar);
        }
        var chip = document.createElement("span");
        chip.className = "chip " + kindClass(op.kind);
        chip.innerHTML = "<span class='k'>" + esc(op.mech) + "</span>" +
          (shortAttrs(op) ? "<span class='a'>" + esc(shortAttrs(op)) + "</span>" : "");
        chip.onclick = function () {
          Array.prototype.forEach.call(box.querySelectorAll(".chip.active"),
            function (c) { c.classList.remove("active"); });
          chip.classList.add("active");
          showProps(L, op);
        };
        chain.appendChild(chip);
      });
      row.appendChild(no);
      row.appendChild(chain);
      box.appendChild(row);
    });
    if (!(m.layers || []).length) {
      box.innerHTML = "<p class='bad'>这份模型一层都没有 —— 先查 schedule 是否被 stack 引用（见 by1attrs A13）。</p>";
    }
  }

  function showProps(L, op) {
    $("props-empty").hidden = true;
    var p = $("props");
    p.hidden = false;
    var rows = Object.keys(op.attrs || {}).sort().map(function (k) {
      var v = op.attrs[k];
      return "<tr><td>" + esc(k) + "</td><td>" + esc(v === null ? "null" : v) + "</td></tr>";
    }).join("");
    p.innerHTML =
      "<h3>" + esc(op.mech) + " <span class='pmeta'>" + esc(op.kind) +
      " · 第 " + L.index + " 层</span></h3>" +
      "<div class='pmeta'>in: " + esc((op.inputs || []).join(", ")) +
      "<br>out: " + esc((op.outputs || []).join(", ")) + "</div>" +
      "<table class='attrs'>" + rows + "</table>";
  }

  $("q").addEventListener("input", function (e) {
    filter = e.target.value;
    renderList();
  });

  $("stats").textContent = INDEX.length + " 份模型";
  renderList();
})();
