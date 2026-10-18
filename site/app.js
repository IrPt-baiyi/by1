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
    $("tabs").hidden = false;
    showTab("struct");
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


  /* ---- 图（M3）---------------------------------------------------- */
  var tab = "struct";

  function showTab(name) {
    tab = name;
    var box = $("tabs");
    if (!box) return;
    Array.prototype.forEach.call(box.querySelectorAll(".tab"), function (b) {
      b.classList.toggle("active", b.getAttribute("data-tab") === name);
    });
    $("props").hidden = true;
    $("props-empty").hidden = false;
    if (name === "diff") {
      renderDiffList();
      return;
    }
    if (!current) return;
    if (name === "graph") {
      renderGraph(current);
    } else {
      var m = window.BY1_MODELS[current];
      if (m) renderLayers(m);
    }
  }

  function loadGraph(slug) {
    if (window.BY1_GRAPH && window.BY1_GRAPH[slug]) {
      return Promise.resolve(window.BY1_GRAPH[slug]);
    }
    return loadScript("data/" + slug + ".graph.js").then(function () {
      return (window.BY1_GRAPH || {})[slug];
    });
  }

  function renderGraph(slug) {
    var box = $("layers");
    box.innerHTML = "<p class='empty small'>正在导图…</p>";
    loadGraph(slug).then(function (g) {
      if (!g) {
        box.innerHTML = "<p class='bad'>这份模型没有图数据（导出时这一步被跳过）。</p>";
        return;
      }
      box.innerHTML = "";
      var sum = document.createElement("div");
      sum.className = "gsum";
      sum.innerHTML = "<span>图 sha <b>" + esc(g.sha) + "</b></span>" +
        (g.golden
          ? "<span>golden " + (g.match ? "✅ 一致" : "❌ 不同") + "</span>"
          : "<span>golden 里没有这份</span>") +
        "<span>节点 <b>" + g.node_count + "</b></span>" +
        "<span>原语 <b>" + Object.keys(g.prims).length + "</b> 种</span>" +
        (g.nodes ? "<span class='dim'>节点表已导出（可点层看属性）</span>"
                 : "<span class='dim'>大模型只导摘要</span>");
      box.appendChild(sum);

      var hist = document.createElement("div");
      hist.className = "prims";
      Object.keys(g.prims).sort().forEach(function (p) {
        var s = document.createElement("span");
        s.className = "pchip";
        s.textContent = p + " × " + g.prims[p];
        hist.appendChild(s);
      });
      box.appendChild(hist);

      (g.layers || []).forEach(function (L) {
        var row = document.createElement("div");
        row.className = "layer";
        var no = document.createElement("div");
        no.className = "lno";
        no.textContent = L.index < 0 ? "全局" : "L" + L.index;
        var chain = document.createElement("div");
        chain.className = "chain";
        var txt = document.createElement("span");
        txt.className = "gchain";
        txt.textContent = L.prims.join(" → ");
        chain.appendChild(txt);
        chain.style.cursor = "pointer";
        chain.onclick = function () { showGraphLayer(g, L); };
        row.appendChild(no);
        row.appendChild(chain);
        box.appendChild(row);
      });
    }).catch(function (e) {
      box.innerHTML = "<p class='bad'>" + esc(e.message) + "</p>";
    });
  }

  function showGraphLayer(g, L) {
    $("props-empty").hidden = true;
    var p = $("props");
    p.hidden = false;
    if (!g.nodes) {
      p.innerHTML = "<h3>" + (L.index < 0 ? "全局" : "第 " + L.index + " 层") + "</h3>" +
        "<div class='pmeta'>这份模型只导了图摘要；节点表在节点 <= 600 的模型上有。</div>" +
        "<div class='sect'>原语链</div><div class='gchain'>" + esc(L.prims.join(" → ")) + "</div>";
      return;
    }
    var prefix = L.index < 0 ? null : ("L" + L.index);
    var rows = (g.nodes || []).filter(function (n) {
      if (!prefix) return !/^L\d/.test(n.outs.join(" "));
      return n.outs.join(" ").indexOf(prefix) >= 0;
    }).map(function (n) {
      var attrs = Object.keys(n.attrs || {}).sort().map(function (k) {
        return esc(k) + "=" + esc(n.attrs[k]);
      }).join("  ");
      return "<div style='margin:6px 0'><b>" + esc(n.prim) + "</b><div class='gchain'>" +
        esc(attrs || "—") + "</div></div>";
    }).join("");
    p.innerHTML = "<h3>" + (L.index < 0 ? "全局" : "第 " + L.index + " 层") + "</h3>" +
      "<div class='pmeta'>" + (L.prims.length) + " 个节点</div>" + rows;
  }


  /* ---- 对比（M4）-------------------------------------------------- */
  var diffIndex = null;

  function loadDiffIndex() {
    if (window.BY1_DIFF_INDEX) {
      return Promise.resolve(window.BY1_DIFF_INDEX);
    }
    return loadScript("data/diff/index.js").then(function () {
      return window.BY1_DIFF_INDEX || [];
    });
  }

  function renderDiffList() {
    var box = $("layers");
    box.innerHTML = "<p class='empty small'>正在加载对比清单…</p>";
    loadDiffIndex().then(function (idx) {
      box.innerHTML = "";
      if (!idx.length) {
        box.innerHTML = "<p class='empty small'>没有预置的对比（python tools/by1site.py --diffs）。</p>";
        return;
      }
      var t = document.createElement("div");
      t.className = "gsum";
      t.innerHTML = "<span>预置对比 <b>" + idx.length + "</b> 对</span>" +
        "<span class='dim'>差异全部由 IR 现算，页面只展示</span>";
      box.appendChild(t);
      idx.forEach(function (d) {
        var row = document.createElement("div");
        row.className = "layer";
        var no = document.createElement("div");
        no.className = "lno";
        no.textContent = "对比";
        var chain = document.createElement("div");
        chain.className = "chain";
        chain.style.cursor = "pointer";
        chain.innerHTML = "<span class='k'>" + esc(d.a) + "</span>" +
          "<span class='arrow'>vs</span>" +
          "<span class='k'>" + esc(d.b) + "</span>" +
          "<span class='a'>层差 " + d.layer_diff + " · 属性差 " + d.attr_changed + "</span>";
        chain.onclick = function () { renderDiff(d); };
        row.appendChild(no);
        row.appendChild(chain);
        box.appendChild(row);
      });
    }).catch(function (e) {
      box.innerHTML = "<p class='bad'>" + esc(e.message) + "</p>";
    });
  }

  function renderDiff(entry) {
    var slug = entry.a_slug + "__" + entry.b_slug;
    var go = (window.BY1_DIFF && window.BY1_DIFF[slug])
      ? Promise.resolve(window.BY1_DIFF[slug])
      : loadScript("data/diff/" + slug + ".js").then(function () {
          return (window.BY1_DIFF || {})[slug];
        });
    go.then(function (d) {
      if (!d) { return; }
      var box = $("layers");
      box.innerHTML = "";
      var sum = document.createElement("div");
      sum.className = "gsum";
      sum.innerHTML = "<span><b>" + esc(d.a) + "</b> vs <b>" + esc(d.b) + "</b></span>" +
        "<span>层差 <b>" + d.counts.layer_diff + "</b></span>" +
        "<span>属性改 <b>" + d.counts.attr_changed + "</b></span>" +
        "<span>仅 A <b>" + d.counts.attr_only_a + "</b> · 仅 B <b>" +
        d.counts.attr_only_b + "</b></span>";
      box.appendChild(sum);

      var ft = document.createElement("table");
      ft.className = "attrs";
      ft.innerHTML = d.fields.map(function (f) {
        return "<tr><td>" + esc(f.key) + "</td><td>" +
          (f.same
            ? "<span class='dim'>" + esc(f.a) + "（同）</span>"
            : esc(f.a) + " <span class='arrow'>→</span> " + esc(f.b)) + "</td></tr>";
      }).join("");
      box.appendChild(ft);

      if ((d.layer_diffs || []).length) {
        var h = document.createElement("div");
        h.className = "sect";
        h.textContent = "有差的层";
        box.appendChild(h);
        d.layer_diffs.slice(0, 60).forEach(function (L) {
          var row = document.createElement("div");
          row.className = "layer";
          var no = document.createElement("div");
          no.className = "lno";
          no.textContent = "L" + L.layer;
          var chain = document.createElement("div");
          chain.className = "chain";
          chain.innerHTML = "<span class='gchain'>A: " +
            esc((L.a || ["（没有这一层）"]).join(" → ")) + "<br>B: " +
            esc((L.b || ["（没有这一层）"]).join(" → ")) + "</span>";
          row.appendChild(no);
          row.appendChild(chain);
          box.appendChild(row);
        });
      }
      if ((d.attr_changed || []).length) {
        var h2 = document.createElement("div");
        h2.className = "sect";
        h2.textContent = "属性差（前 " + Math.min(200, d.attr_changed.length) + " 条）";
        box.appendChild(h2);
        var at = document.createElement("table");
        at.className = "attrs";
        at.innerHTML = d.attr_changed.map(function (c) {
          return "<tr><td>" + esc(c.k) + "</td><td>" + esc(c.a) +
            " <span class='arrow'>→</span> " + esc(c.b) + "</td></tr>";
        }).join("");
        box.appendChild(at);
      }
    }).catch(function (e) {
      $("layers").innerHTML = "<p class='bad'>" + esc(e.message) + "</p>";
    });
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

  Array.prototype.forEach.call($("tabs").querySelectorAll(".tab"), function (b) {
    b.onclick = function () { showTab(b.getAttribute("data-tab")); };
  });

  $("stats").textContent = INDEX.length + " 份模型";
  renderList();
})();
