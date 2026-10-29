/* Pulse AI: chat UI. Talks to server.py over JSON + Server-Sent Events. */
(() => {
  "use strict";

  // ---------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem("pulse." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("pulse." + k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const ICON = {
    bolt: '<svg viewBox="0 0 24 24"><path d="M13.5 2 4 14h6.5L9 22l10-12.5h-6.5z"/></svg>',
    copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>',
    check: '<svg viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
    x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
    retry: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>',
    pencil: '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
    trash: '<svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>',
    globe: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/></svg>',
    play: '<svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>',
    key: '<svg viewBox="0 0 24 24"><circle cx="7.5" cy="15.5" r="4.5"/><path d="m10.7 12.3 9.3-9.3M17 6l3 3M14 9l2 2"/></svg>',
  };
  const STARTERS = {
    chat: [
      { ico: "📈", t: "Why is NVDA moving?", d: "Live quote + today's company news", p: "Why is NVDA stock moving today?" },
      { ico: "🌏", t: "Markets right now", d: "Nifty, Sensex, S&P 500, gold and oil", p: "How are Nifty, Sensex, S&P 500, gold and oil doing right now?" },
      { ico: "🐦", t: "What experts are posting", d: "Latest AI posts on X, Bluesky, Mastodon", p: "What are AI researchers posting about on social media today?" },
      { ico: "📰", t: "Today's top headlines", d: "Breaking news with sources", p: "What are the top news stories in the world today?" },
      { ico: "🌦️", t: "Weather & air quality", d: "Forecast and AQI for your city", p: "Weather and air quality in Hyderabad this week" },
      { ico: "⚽", t: "Live scores", d: "Premier League, NBA, F1 and more", p: "Live Premier League scores and today's fixtures" },
    ],
    code: [
      { ico: "⚡", t: "Build an API", d: "FastAPI endpoint, tested", p: "Write a FastAPI app with a /todos CRUD API using an in-memory list. Include Pydantic models and show how to run it." },
      { ico: "🐛", t: "Fix a bug", d: "Paste an error or broken code", p: "Why does this fail and how do I fix it?\n\nimport json\ndata = json.loads(\"{'name': 'Ana'}\")\nprint(data['name'])" },
      { ico: "📦", t: "Latest versions", d: "Live from PyPI and npm", p: "What are the latest versions of pandas, fastapi and react, and when were they released?" },
      { ico: "📂", t: "Explain a repo", d: "Reads real code on GitHub", p: "Explain how the repo tiangolo/typer is structured and what its main entry point does." },
      { ico: "🧪", t: "Write and test", d: "Runs the code to verify", p: "Write a Python function that checks if a string is a valid IPv4 address, with tests, and run the tests." },
      { ico: "🔍", t: "Review my code", d: "Bugs, security, style", p: "Review this code for bugs and security issues:\n\nimport sqlite3\ndef get_user(name):\n    conn = sqlite3.connect('app.db')\n    return conn.execute(f\"SELECT * FROM users WHERE name = '{name}'\").fetchall()" },
    ],
  };
  const MODEL_HINTS = {
    "openai/gpt-oss-120b": "Best quality on Groq · free tier",
    "openai/gpt-oss-20b": "Faster · good for quick lookups",
    "qwen/qwen3.8-27b": "Multilingual",
    "claude-opus-5-5": "Most capable · best for hard coding tasks",
    "claude-sonnet-5-5": "Fast and smart · everyday coding",
    "claude-haiku-5-5": "Fastest and cheapest Claude",
    "claude-fable-5-1": "Anthropic's most capable model · premium",
  };
  const PROVIDER_SHORT = { groq: "Groq", anthropic: "Claude", openai: "OpenAI" };
  const MENU_LIMIT = 8;

  // ---------------------------------------------------------------- state
  const legacyKey = store.get("apiKey", "");
  const state = {
    config: null,
    conversations: [],
    currentId: null,
    controller: null,
    userModels: store.get("userModels", {}), // models fetched for keys entered in Settings
    expanded: {},
    attachments: [], // [{key, kind: "doc"|"image", name, size, status, id?, dataUrl?, error?}]
    auth: { client: null, user: null },
    settings: {
      apiKeys: store.get("apiKeys", legacyKey ? { groq: legacyKey } : {}),
      userId: store.get("userId", "default"),
      tz: store.get("tz", Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"),
      model: store.get("model", null),
      mode: store.get("mode", "chat"),
      tools: store.get("tools", null), // null = all
    },
  };

  const ui = {
    sidebar: $("#sidebar"), scrim: $("#scrim"), history: $("#history"), search: $("#searchChats"),
    thread: $("#thread"), messages: $("#messages"), welcome: $("#welcome"), starters: $("#starters"),
    input: $("#input"), send: $("#sendBtn"), form: $("#composer"), scrollDown: $("#scrollDown"),
    modelBtn: $("#modelBtn"), modelMenu: $("#modelMenu"), modelName: $("#modelName"),
    settings: $("#settings"), toolGrid: $("#toolGrid"), toast: $("#toast"), keyList: $("#keyList"),
    disclaimer: $("#disclaimer"),
  };

  // ---------------------------------------------------------------- providers & models
  const providers = () => state.config?.providers || [];
  const providerOf = (value) => (value || "").split(":")[0];
  const modelOf = (value) => (value || "").split(":").slice(1).join(":");
  const hasKey = (p) => !!(p.server_key || state.settings.apiKeys[p.id]);
  const modelsFor = (p) => (p.server_key ? p.models : state.userModels[p.id] || []);
  const prettyModel = (value) => {
    if (!value) return "";
    if (!value.includes(":")) value = "groq:" + value;
    return `${PROVIDER_SHORT[providerOf(value)] || providerOf(value)} · ${modelOf(value)}`;
  };
  function modelAvailable(value) {
    const p = providers().find((x) => x.id === providerOf(value));
    return !!p && hasKey(p) && modelsFor(p).includes(modelOf(value));
  }
  function setModel(value) {
    state.settings.model = value; store.set("model", value);
    ui.modelName.textContent = prettyModel(value);
    renderModelMenu(); renderDisclaimer();
  }

  // ---------------------------------------------------------------- markdown
  if (window.marked) marked.setOptions({ gfm: true, breaks: false });

  // Some models (gpt-oss) emit citation markup like 【1】, 【tool】 or 【name](url): normalise or drop it.
  function cleanCitations(text) {
    return (text || "")
      // non-breaking hyphens/spaces look normal but break copy-pasted codes, commands and numbers
      .replace(/[‐‑]/g, "-").replace(/[  ]/g, " ")
      .replace(/【([^】\]\n]*)\]\((https?:\/\/[^)\s]+)\)/g, "[$1]($2)")
      // keep file citations readable, e.g. 【report.pdf, section 3】 -> (report.pdf, section 3)
      .replace(/【([^】\n]*\.\w{2,5}\b[^】\n]*)】/g, " *($1)*")
      .replace(/【[^】\n]*】/g, "")
      .replace(/\[([^\]\n]*)\]\(https?:\/\/api\.openai\.com[^)]*\)/g, "$1")
      .replace(/\s*\[(?:link|source)\]\(#?\)/gi, "")
      .replace(/\[([^\]\n]*)\]\(#?\)/g, "$1");
  }

  function renderMarkdown(target, text, { final = false } = {}) {
    text = cleanCitations(text);
    let html;
    if (window.marked && window.DOMPurify) {
      html = DOMPurify.sanitize(marked.parse(text || ""), { ADD_ATTR: ["target", "rel"] });
    } else {
      html = "<p>" + esc(text).replace(/\n/g, "<br>") + "</p>";
    }
    target.innerHTML = html;
    target.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
    target.querySelectorAll("table").forEach((t) => {
      if (t.parentElement.classList.contains("table-wrap")) return;
      const w = el("div", "table-wrap"); t.replaceWith(w); w.appendChild(t);
    });
    target.querySelectorAll("pre > code").forEach((code) => {
      const pre = code.parentElement;
      const lang = ((code.className.match(/language-([\w+-]+)/) || [])[1] || "text").toLowerCase();
      if (window.hljs) { try { hljs.highlightElement(code); } catch { /* unknown lang */ } }
      const head = el("div", "code-head", `<span>${esc(lang)}</span><span class="code-actions"></span>`);
      const actions = $(".code-actions", head);
      // Run button for finished Python blocks, when the server allows code execution
      if (final && state.config?.code_execution && ["python", "py", "python3"].includes(lang)) {
        const run = el("button", "run-btn", `${ICON.play}Run`);
        run.type = "button";
        run.onclick = () => runCode(code.innerText, pre, run);
        actions.append(run);
      }
      const copy = el("button", "", `${ICON.copy}Copy`);
      copy.type = "button";
      copy.onclick = () => copyText(code.innerText, copy);
      actions.append(copy);
      pre.prepend(head);
    });
  }

  async function runCode(code, pre, btn) {
    btn.disabled = true; btn.innerHTML = `<span class="spin"></span>Running`;
    let out = pre.nextElementSibling?.classList.contains("run-output") ? pre.nextElementSibling : null;
    if (!out) { out = el("div", "run-output"); pre.after(out); }
    out.innerHTML = '<div class="run-head">Running…</div>';
    try {
      const r = await api("/api/run", { method: "POST", body: JSON.stringify({ code }) });
      const ok = r.exit_code === 0;
      out.className = "run-output " + (ok ? "ok" : "fail");
      out.innerHTML = `<div class="run-head">${ok ? "✓ Ran" : r.exit_code == null ? "⏱ Timed out" : `✗ Exit code ${r.exit_code}`} · ${(r.duration_ms / 1000).toFixed(2)}s</div>`
        + (r.stdout ? `<pre>${esc(r.stdout)}</pre>` : "")
        + (r.stderr ? `<pre class="stderr">${esc(r.stderr)}</pre>` : "")
        + (!r.stdout && !r.stderr ? '<pre class="muted">(no output)</pre>' : "");
    } catch (e) {
      out.className = "run-output fail";
      out.innerHTML = `<div class="run-head">✗ ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false; btn.innerHTML = `${ICON.play}Run`;
    }
  }

  // ---------------------------------------------------------------- utils
  function toast(msg, ms = 2800) {
    ui.toast.textContent = msg; ui.toast.hidden = false;
    clearTimeout(toast.t); toast.t = setTimeout(() => (ui.toast.hidden = true), ms);
  }
  async function copyText(text, btn) {
    try { await navigator.clipboard.writeText(text); } catch { return toast("Couldn't copy"); }
    if (btn) { const old = btn.innerHTML; btn.innerHTML = ICON.check + (old.includes("Copy") ? "Copied" : ""); setTimeout(() => (btn.innerHTML = old), 1400); }
  }
  // Every request carries the signed-in user's token (when login is on) and the user id (when it's off)
  async function authHeaders() {
    if (!state.auth.client) return {};
    const { data } = await state.auth.client.auth.getSession();
    return data.session ? { Authorization: `Bearer ${data.session.access_token}` } : {};
  }
  const withUser = (path) => path + (path.includes("?") ? "&" : "?") + "user_id=" + encodeURIComponent(state.settings.userId);
  async function api(path, opts = {}) {
    const headers = { ...(opts.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...(await authHeaders()), ...(opts.headers || {}) };
    const res = await fetch(withUser(path), { ...opts, headers });
    if (res.status === 401 && state.auth.client) { showLogin("Your session expired. Please sign in again."); throw new Error("Please sign in."); }
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    return res.json();
  }
  const nearBottom = () => ui.thread.scrollHeight - ui.thread.scrollTop - ui.thread.clientHeight < 120;
  const scrollToBottom = (force) => { if (force || nearBottom()) ui.thread.scrollTop = ui.thread.scrollHeight; };
  const fmtArgs = (args) => Object.values(args || {}).filter((v) => typeof v === "string" && v)
    .map((v) => (v.length > 60 ? v.slice(0, 57) + "…" : v)).join(" · ");

  // ---------------------------------------------------------------- sidebar / history
  function groupLabel(iso) {
    const d = new Date(iso), now = new Date();
    const days = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / 864e5);
    if (days <= 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return "Previous 7 days";
    if (days < 30) return "Previous 30 days";
    return d.toLocaleString(undefined, { month: "long", year: "numeric" });
  }

  function renderHistory() {
    const q = ui.search.value.trim().toLowerCase();
    const list = state.conversations.filter((c) => !q || c.title.toLowerCase().includes(q));
    ui.history.innerHTML = "";
    if (!list.length) {
      ui.history.append(el("div", "history-empty", q ? "No chats match your search." : "Your chats will appear here."));
      return;
    }
    let group = null, groupEl = null;
    for (const c of list) {
      const label = groupLabel(c.updated_at);
      if (label !== group) {
        group = label; groupEl = el("div", "history-group");
        groupEl.append(el("div", "history-label", esc(label))); ui.history.append(groupEl);
      }
      const item = el("div", "conv" + (c.id === state.currentId ? " active" : ""));
      item.innerHTML = `<span class="title">${esc(c.title)}</span>
        <span class="actions"><button class="ren" title="Rename">${ICON.pencil}</button><button class="del" title="Delete">${ICON.trash}</button></span>`;
      item.onclick = (e) => { if (!e.target.closest(".actions") && !e.target.closest("input")) openConversation(c.id); };
      item.querySelector(".ren").onclick = () => startRename(item, c);
      item.querySelector(".del").onclick = () => deleteConversation(c);
      groupEl.append(item);
    }
  }

  async function loadHistory() {
    try { state.conversations = await api("/api/conversations"); }
    catch (e) { state.conversations = []; toast("Couldn't load chat history: " + e.message); }
    renderHistory();
  }

  function startRename(item, c) {
    const span = item.querySelector(".title");
    span.innerHTML = `<input value="${esc(c.title)}" maxlength="120" />`;
    const input = span.querySelector("input");
    input.focus(); input.select();
    const finish = async (save) => {
      const title = input.value.trim();
      if (save && title && title !== c.title) {
        c.title = title;
        try { await api(`/api/conversations/${c.id}`, { method: "PATCH", body: JSON.stringify({ title }) }); } catch (e) { toast(e.message); }
      }
      renderHistory();
    };
    input.onkeydown = (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
    input.onblur = () => finish(true);
  }

  async function deleteConversation(c) {
    if (!confirm(`Delete "${c.title}"?`)) return;
    try { await api(`/api/conversations/${c.id}`, { method: "DELETE" }); } catch (e) { return toast(e.message); }
    state.conversations = state.conversations.filter((x) => x.id !== c.id);
    if (c.id === state.currentId) newChat(); else renderHistory();
  }

  function setSidebar(open) { ui.sidebar.classList.toggle("open", open); ui.scrim.classList.toggle("show", open); }

  // ---------------------------------------------------------------- conversations
  function newChat() {
    stopStreaming();
    state.currentId = null;
    history.replaceState(null, "", location.pathname);
    ui.messages.innerHTML = "";
    ui.welcome.hidden = false;
    renderHistory(); setSidebar(false); ui.input.focus();
  }

  async function openConversation(id) {
    stopStreaming();
    state.currentId = id; renderHistory(); setSidebar(false);
    history.replaceState(null, "", "#c=" + id);
    ui.welcome.hidden = true; ui.messages.innerHTML = "";
    try {
      const msgs = await api(`/api/conversations/${id}/messages`);
      for (const m of msgs) {
        if (m.role === "user") addUserMessage(m.content, (m.metadata && m.metadata.files) || []);
        else {
          const view = addAssistantMessage();
          view.thinking.remove();
          view.text = m.content;
          renderMarkdown(view.content, m.content || "_(stopped)_", { final: true });
          finishAssistant(view, m.metadata || {});
        }
      }
      scrollToBottom(true);
    } catch (e) { toast("Couldn't open chat: " + e.message); }
  }

  // ---------------------------------------------------------------- messages
  function addUserMessage(text, files = []) {
    const m = el("div", "msg user");
    const wrap = el("div", "user-wrap");
    if (files.length) {
      const chips = el("div", "msg-files");
      for (const f of files) {
        const chip = el("span", "file-chip sent", f.thumb ? `<img src="${esc(f.thumb)}" alt="">` : fileIcon(f.name));
        chip.append(el("span", "fname", esc(f.name)));
        chips.append(chip);
      }
      wrap.append(chips);
    }
    const b = el("div", "bubble"); b.textContent = text;
    wrap.append(b); m.append(wrap); ui.messages.append(m);
    return m;
  }

  function fileIcon(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    const map = { pdf: "📕", docx: "📘", md: "📝", txt: "📄", csv: "📊", tsv: "📊", json: "🧾", html: "🌐", htm: "🌐",
      png: "🖼️", jpg: "🖼️", jpeg: "🖼️", webp: "🖼️", gif: "🖼️" };
    return `<span class="ficon">${map[ext] || "💻"}</span>`;
  }

  function addAssistantMessage() {
    const m = el("div", "msg assistant");
    m.innerHTML = `<div class="avatar bot">${ICON.bolt}</div>
      <div class="body"><div class="steps"></div><div class="thinking"><span></span><span></span><span></span></div><div class="md"></div><div class="foot-slot"></div></div>`;
    ui.messages.append(m);
    return { root: m, steps: $(".steps", m), thinking: $(".thinking", m), content: $(".md", m), foot: $(".foot-slot", m), text: "", stepEls: {} };
  }

  function addStep(view, ev) {
    const s = el("span", "step", `<span class="spin"></span><span>${esc(ev.label)}</span><span class="q">${esc(fmtArgs(ev.args))}</span>`);
    s.title = JSON.stringify(ev.args).slice(0, 500);
    view.steps.append(s); view.stepEls[ev.id] = s;
  }

  function finishStep(view, ev) {
    const s = view.stepEls[ev.id]; if (!s) return;
    s.classList.add("done");
    s.querySelector(".spin").outerHTML = ev.ok ? `<span class="ok">${ICON.check}</span>` : `<span class="fail">${ICON.x}</span>`;
    s.append(el("span", "ms", `${(ev.latency_ms / 1000).toFixed(1)}s`));
  }

  function finishAssistant(view, meta) {
    view.content.classList.remove("cursor");
    view.thinking.remove();
    // keep live tool chips only while streaming; the sources panel replaces them
    view.steps.innerHTML = "";
    const tools = meta.tools || [];
    const foot = el("div", "msg-foot");
    const copyBtn = el("button", "", ICON.copy); copyBtn.title = "Copy"; copyBtn.onclick = () => copyText(cleanCitations(view.text), copyBtn);
    foot.append(copyBtn);
    if (view.retryText) {
      const r = el("button", "", ICON.retry); r.title = "Regenerate"; r.onclick = () => sendMessage(view.retryText);
      foot.append(r);
    }
    let panel = null;
    if (tools.length) {
      const t = el("button", "sources-toggle", `${ICON.globe}${tools.length} ${meta.mode === "code" ? "step" : "source"}${tools.length > 1 ? "s" : ""}`);
      t.onclick = () => { panel.hidden = !panel.hidden; };
      foot.append(t);
      panel = el("div", "sources"); panel.hidden = true;
      for (const tool of tools) {
        const label = (state.config?.tools.find((x) => x.name === tool.name) || {}).label
          || (tool.name === "search_files" ? "📎 Your files" : tool.name);
        panel.append(el("div", "src", `<span>${esc(label)}</span><span class="args">${esc(fmtArgs(tool.args))}</span><span class="ms">${((tool.latency_ms || 0) / 1000).toFixed(1)}s</span>`));
      }
    }
    const metaParts = [];
    if (meta.mode === "code") metaParts.push('<span class="mode-tag">Code</span>');
    if (meta.model) metaParts.push(esc(prettyModel(meta.model)));
    if (meta.total_ms) metaParts.push(`${(meta.total_ms / 1000).toFixed(1)}s`);
    if (meta.status === "stopped") metaParts.push("stopped");
    if (metaParts.length) foot.append(el("span", "meta", metaParts.join(" · ")));
    view.foot.replaceChildren(foot);
    if (panel) view.foot.append(panel);
  }

  // ---------------------------------------------------------------- streaming chat
  function setStreaming(on) {
    ui.send.classList.toggle("stop", on);
    ui.send.disabled = !on && !ui.input.value.trim();
    ui.send.setAttribute("aria-label", on ? "Stop" : "Send");
  }
  function stopStreaming() { if (state.controller) { state.controller.abort(); state.controller = null; setStreaming(false); } }

  async function sendMessage(text) {
    text = text.trim();
    if (!text && state.attachments.some((a) => a.status === "ready")) {
      text = state.attachments.some((a) => a.kind === "doc") ? "Summarize the attached file(s)." : "Describe this image.";
    }
    if (!text || state.controller) return;
    if (!modelAvailable(state.settings.model)) {
      const p = providers().find((x) => x.id === providerOf(state.settings.model));
      openSettings(p?.id);
      toast(p && !hasKey(p) ? `Add your ${p.name} API key, or pick another model` : "Pick a model at the top of the page");
      return;
    }
    if (state.attachments.some((a) => a.status === "uploading")) { toast("Wait for your files to finish uploading"); return; }
    const ready = state.attachments.filter((a) => a.status === "ready");
    const fileIds = ready.filter((a) => a.kind === "doc").map((a) => a.id);
    const images = ready.filter((a) => a.kind === "image").map((a) => ({ name: a.name, data_url: a.dataUrl }));
    const sentFiles = ready.map((a) => ({ name: a.name, thumb: a.kind === "image" ? a.dataUrl : null }));
    state.attachments = []; renderAttachments();

    ui.welcome.hidden = true;
    ui.input.value = ""; autoresize();
    addUserMessage(text, sentFiles);
    const view = addAssistantMessage();
    view.retryText = text;
    scrollToBottom(true);

    const controller = new AbortController();
    state.controller = controller; setStreaming(true);
    let meta = {}, raf = 0;
    const paint = () => { raf = 0; renderMarkdown(view.content, view.text); view.content.classList.add("cursor"); scrollToBottom(); };

    try {
      const res = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json", ...(await authHeaders()) }, signal: controller.signal,
        body: JSON.stringify({
          message: text, conversation_id: state.currentId, user_id: state.settings.userId,
          model: state.settings.model, mode: state.settings.mode, tools: state.settings.tools,
          user_tz: state.settings.tz, api_keys: state.settings.apiKeys, file_ids: fileIds, images,
        }),
      });
      if (res.status === 401 && state.auth.client) showLogin("Your session expired. Please sign in again.");
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const event = (block.match(/^event: (.*)$/m) || [])[1];
          const dataLine = (block.match(/^data: (.*)$/m) || [])[1];
          if (!event || dataLine == null) continue;
          const data = JSON.parse(dataLine);
          switch (event) {
            case "meta":
              if (state.currentId !== data.conversation_id) {
                state.currentId = data.conversation_id;
                history.replaceState(null, "", "#c=" + data.conversation_id);
                loadHistory();
              }
              break;
            case "token":
              view.thinking.hidden = true; view.text += data.text;
              if (!raf) raf = requestAnimationFrame(paint);
              break;
            case "reset":
              view.text = ""; view.content.innerHTML = ""; view.thinking.hidden = false;
              break;
            case "tool_start": addStep(view, data); scrollToBottom(); break;
            case "notice": view.steps.append(el("span", "step", `<span>⚡ ${esc(data.message)}</span>`)); break;
            case "tool_end": finishStep(view, data); break;
            case "error":
              view.thinking.hidden = true;
              view.content.after(el("div", "err", esc(data.message)));
              break;
            case "done": meta = data; break;
          }
        }
      }
    } catch (e) {
      if (e.name === "AbortError") meta = { ...meta, status: "stopped" };
      else { view.thinking.hidden = true; view.content.after(el("div", "err", esc(e.message))); }
    } finally {
      if (raf) cancelAnimationFrame(raf);
      if (view.text) renderMarkdown(view.content, view.text, { final: true });
      else if (meta.status === "stopped") renderMarkdown(view.content, "_Stopped._");
      if (state.controller === controller) state.controller = null;
      setStreaming(false);
      finishAssistant(view, meta);
      scrollToBottom();
      loadHistory();
    }
  }

  // ---------------------------------------------------------------- attachments
  const IMAGE_RE = /\.(png|jpe?g|webp|gif)$/i;
  const fmtSize = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);

  function addFiles(fileList) {
    const maxMb = state.config?.upload?.max_mb || 20;
    for (const file of fileList) {
      if (state.attachments.length >= 10) { toast("Up to 10 files per message"); break; }
      const isImage = file.type.startsWith("image/") || IMAGE_RE.test(file.name);
      const att = { key: Math.random().toString(36).slice(2), kind: isImage ? "image" : "doc", name: file.name || "pasted-image.png", size: file.size, status: "uploading" };
      if (file.size > maxMb * 1048576) { att.status = "error"; att.error = `Larger than ${maxMb} MB`; }
      state.attachments.push(att);
      if (att.status === "uploading") (isImage ? readImage(att, file) : uploadDoc(att, file));
    }
    renderAttachments();
  }

  async function uploadDoc(att, file) {
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const r = await api("/api/files", { method: "POST", body: form });
      Object.assign(att, { status: "ready", id: r.id, detail: `${fmtSize(att.size)} · ${r.n_chunks} section${r.n_chunks === 1 ? "" : "s"}` });
    } catch (e) { Object.assign(att, { status: "error", error: e.message }); }
    renderAttachments();
  }

  // Images are sent to vision models directly; shrink them so requests stay small and fast
  function readImage(att, file) {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, 1600 / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(img.width * scale); canvas.height = Math.round(img.height * scale);
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        Object.assign(att, { status: "ready", dataUrl: canvas.toDataURL("image/jpeg", 0.85), detail: `${img.width}×${img.height}` });
        renderAttachments();
      };
      img.onerror = () => { Object.assign(att, { status: "error", error: "Not a readable image" }); renderAttachments(); };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

  function renderAttachments() {
    const box = $("#attachments");
    box.innerHTML = "";
    box.hidden = !state.attachments.length;
    for (const a of state.attachments) {
      const chip = el("div", `file-chip ${a.status}`, a.kind === "image" && a.dataUrl ? `<img src="${a.dataUrl}" alt="">` : fileIcon(a.name));
      const info = el("span", "finfo", `<span class="fname">${esc(a.name)}</span><span class="fdetail">${
        a.status === "uploading" ? (a.kind === "image" ? "Preparing…" : "Reading & indexing…") : a.status === "error" ? esc(a.error) : esc(a.detail || "")}</span>`);
      chip.append(info);
      if (a.status === "uploading") chip.append(el("span", "spin"));
      const rm = el("button", "frm", ICON.x); rm.type = "button"; rm.title = "Remove";
      rm.onclick = () => {
        state.attachments = state.attachments.filter((x) => x !== a);
        if (a.id) api(`/api/files/${a.id}`, { method: "DELETE" }).catch(() => {});
        renderAttachments();
      };
      chip.append(rm);
      box.append(chip);
    }
    if (!state.controller) ui.send.disabled = !ui.input.value.trim() && !state.attachments.some((a) => a.status === "ready");
  }

  // ---------------------------------------------------------------- login (Supabase Auth, optional)
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script"); s.src = src; s.onload = resolve; s.onerror = () => reject(new Error("Couldn't load " + src));
      document.head.append(s);
    });
  }

  async function initAuth() {
    const cfg = state.config.auth;
    if (!cfg?.enabled) return true;
    await loadScript("https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.45.4/dist/umd/supabase.min.js");
    state.auth.client = window.supabase.createClient(cfg.supabase_url, cfg.anon_key);
    state.auth.client.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_OUT") showLogin();
      else if (session?.user) setUser(session.user);
    });
    const { data } = await state.auth.client.auth.getSession();
    if (data.session?.user) { setUser(data.session.user); return true; }
    showLogin();
    return false;
  }

  function setUser(user) {
    state.auth.user = user;
    state.settings.userId = user.id; // server derives the real id from the token; this keeps the UI in sync
    $("#authScreen").hidden = true;
    renderUser();
  }

  function showLogin(message) {
    $("#authScreen").hidden = false;
    $("#authMsg").textContent = message || "";
    $("#authMsg").className = "auth-msg" + (message ? " warn" : "");
    setTimeout(() => $("#authEmail").focus(), 50);
  }

  function bindAuth() {
    let signUp = false;
    const setModeText = () => {
      $("#authTitle").textContent = signUp ? "Create your account" : "Welcome back";
      $("#authSubmit").textContent = signUp ? "Create account" : "Sign in";
      $("#authSwitch").innerHTML = signUp ? 'Already have an account? <button type="button" class="link-btn">Sign in</button>'
        : 'New here? <button type="button" class="link-btn">Create an account</button>';
      $("#authSwitch button").onclick = () => { signUp = !signUp; setModeText(); $("#authMsg").textContent = ""; };
    };
    setModeText();
    $("#authForm").onsubmit = async (e) => {
      e.preventDefault();
      const email = $("#authEmail").value.trim(), password = $("#authPassword").value;
      const btn = $("#authSubmit"); btn.disabled = true;
      const msg = $("#authMsg"); msg.className = "auth-msg"; msg.textContent = signUp ? "Creating account…" : "Signing in…";
      try {
        const sb = state.auth.client.auth;
        const { data, error } = signUp
          ? await sb.signUp({ email, password, options: { emailRedirectTo: location.origin } })
          : await sb.signInWithPassword({ email, password });
        if (error) throw error;
        if (signUp && !data.session) {
          msg.className = "auth-msg ok"; msg.textContent = "Check your inbox to confirm your email, then sign in.";
          signUp = false; setModeText();
        } else {
          setUser(data.user); await loadHistory(); msg.textContent = "";
        }
      } catch (err) {
        msg.className = "auth-msg warn"; msg.textContent = err.message || "Sign-in failed";
      } finally { btn.disabled = false; }
    };
    $("#signOut").onclick = async () => {
      if (!state.auth.client) return;
      await state.auth.client.auth.signOut();
      state.conversations = []; newChat(); renderHistory();
    };
  }

  // ---------------------------------------------------------------- model menu
  function renderModelMenu() {
    ui.modelMenu.innerHTML = "";
    for (const p of providers()) {
      const section = el("div", "menu-section");
      section.append(el("div", "menu-note", `${esc(p.name)}${p.server_key ? "" : state.settings.apiKeys[p.id] ? ' <span class="badge">your key</span>' : ""}`));
      if (!hasKey(p)) {
        const add = el("button", "add-key", `${ICON.key}<span><span>Add ${esc(PROVIDER_SHORT[p.id])} API key</span><span class="hint">${esc(p.hint)}</span></span>`);
        add.onclick = () => { ui.modelMenu.hidden = true; openSettings(p.id); };
        section.append(add);
      } else {
        const models = modelsFor(p);
        if (!models.length) section.append(el("div", "menu-empty", "No models found for this key."));
        const shown = state.expanded[p.id] ? models : models.slice(0, MENU_LIMIT);
        for (const m of shown) {
          const value = `${p.id}:${m}`;
          const hint = MODEL_HINTS[m] || (m.includes("codex") ? "Codex · coding specialist" : `${PROVIDER_SHORT[p.id]} model`);
          const b = el("button", value === state.settings.model ? "selected" : "",
            `<span><span>${esc(m)}</span><span class="hint">${esc(hint)}</span></span><span class="check">${ICON.check}</span>`);
          b.onclick = () => { setModel(value); ui.modelMenu.hidden = true; };
          section.append(b);
        }
        if (models.length > MENU_LIMIT && !state.expanded[p.id]) {
          const more = el("button", "more", `<span class="hint">Show all ${models.length} models</span>`);
          more.onclick = (e) => { e.stopPropagation(); state.expanded[p.id] = true; renderModelMenu(); };
          section.append(more);
        }
      }
      ui.modelMenu.append(section);
    }
  }

  // ---------------------------------------------------------------- mode (Chat / Code)
  function setMode(mode) {
    state.settings.mode = mode; store.set("mode", mode);
    document.body.classList.toggle("code-mode", mode === "code");
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    ui.input.placeholder = mode === "code"
      ? "Describe a coding task, paste code or an error…"
      : "Ask about anything happening right now…";
    renderStarters(); renderUser(); renderDisclaimer();
  }

  function renderDisclaimer() {
    const groq = providerOf(state.settings.model) === "groq";
    if (state.settings.mode === "code" && groq) {
      ui.disclaimer.innerHTML = 'Tip: Claude and Codex models give the best coding results. <button type="button" class="link-btn" id="addKeyTip">Add an API key</button>';
      $("#addKeyTip").onclick = () => openSettings("anthropic");
    } else if (state.settings.mode === "code") {
      ui.disclaimer.textContent = state.config?.code_execution
        ? "Code mode can run Python on this computer to test its answers. Review code before using it."
        : "Review generated code before using it.";
    } else {
      ui.disclaimer.textContent = "Pulse AI fetches live data but can still make mistakes. Check important info at the linked sources.";
    }
  }

  // ---------------------------------------------------------------- settings
  function renderToolGrid() {
    ui.toolGrid.innerHTML = "";
    const enabled = state.settings.tools;
    for (const t of state.config.tools) {
      const lab = el("label", "tool-chip", `<input type="checkbox" value="${esc(t.name)}" ${!enabled || enabled.includes(t.name) ? "checked" : ""}/> ${esc(t.label)}`);
      ui.toolGrid.append(lab);
    }
  }

  function renderKeyList(focus) {
    ui.keyList.innerHTML = "";
    for (const p of providers()) {
      const row = el("label", "key-row");
      const status = p.server_key
        ? '<span class="key-status ok">✓ Set on the server (.env)</span>'
        : state.settings.apiKeys[p.id]
          ? `<span class="key-status ok">✓ Saved in this browser · ${(state.userModels[p.id] || []).length} models</span>`
          : `<span class="key-status">Optional · <a href="${esc(p.key_url)}" target="_blank" rel="noopener">get a key</a></span>`;
      row.innerHTML = `<span class="key-name">${esc(p.name)}</span>
        <input type="password" data-provider="${esc(p.id)}" placeholder="${p.server_key ? "Using the server's key" : "Paste API key"}"
          value="${esc(state.settings.apiKeys[p.id] || "")}" ${p.server_key ? "disabled" : ""} autocomplete="off" />
        ${status}`;
      ui.keyList.append(row);
      if (focus === p.id && !p.server_key) setTimeout(() => row.querySelector("input").focus(), 50);
    }
  }

  function openSettings(focusProvider) {
    $("#userIdInput").value = state.settings.userId;
    $("#tzInput").value = state.settings.tz;
    renderKeyList(focusProvider);
    renderToolGrid();
    ui.settings.showModal();
  }

  async function saveSettings() {
    const prevUser = state.settings.userId;
    state.settings.userId = $("#userIdInput").value.trim() || "default";
    state.settings.tz = $("#tzInput").value.trim() || "UTC";
    const checked = [...ui.toolGrid.querySelectorAll("input:checked")].map((i) => i.value);
    state.settings.tools = checked.length === state.config.tools.length ? null : checked;

    // API keys: validate new/changed keys by loading their model list
    const changes = [];
    for (const input of ui.keyList.querySelectorAll("input[data-provider]:not([disabled])")) {
      const id = input.dataset.provider, key = input.value.trim();
      if (key === (state.settings.apiKeys[id] || "")) continue;
      if (!key) { delete state.settings.apiKeys[id]; delete state.userModels[id]; continue; }
      changes.push((async () => {
        try {
          const { models } = await api("/api/models", { method: "POST", body: JSON.stringify({ provider: id, api_key: key }) });
          state.settings.apiKeys[id] = key; state.userModels[id] = models;
          return { id, ok: true, n: models.length, first: models[0] };
        } catch (e) { return { id, ok: false, msg: e.message }; }
      })());
    }
    const results = await Promise.all(changes);
    for (const k of ["userId", "tz", "tools", "apiKeys"]) store.set(k, state.settings[k]);
    store.set("userModels", state.userModels);

    renderUser();
    const added = results.filter((r) => r.ok);
    const failed = results.filter((r) => !r.ok);
    // switch to a newly added provider's best model if the current one isn't usable
    if (!modelAvailable(state.settings.model) && added.length && added[0].first) setModel(`${added[0].id}:${added[0].first}`);
    else { renderModelMenu(); renderDisclaimer(); }
    if (prevUser !== state.settings.userId) { newChat(); loadHistory(); }
    if (failed.length) toast(`${PROVIDER_SHORT[failed[0].id]} key: ${failed[0].msg}`, 5000);
    else if (added.length) toast(added.map((r) => `${PROVIDER_SHORT[r.id]}: ${r.n} models available`).join(" · "), 4000);
    else toast("Settings saved");
  }

  function renderUser() {
    const email = state.auth.user?.email;
    const name = email ? email.split("@")[0] : state.settings.userId;
    $("#userName").textContent = email || name;
    $("#userAvatar").textContent = (name[0] || "U").toUpperCase();
    $("#signOut").hidden = !state.auth.user;
    $("#nameField").hidden = !!state.auth.client;
    const h = new Date().getHours();
    const part = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
    const named = name && name !== "default";
    $("#greeting").textContent = state.settings.mode === "code"
      ? (named ? `${part}, ${name}. What are we building?` : "What are we building today?")
      : (named ? `${part}, ${name}. What's happening?` : "What's happening right now?");
    if (state.config) {
      $("#sourcesCount").textContent = state.settings.mode === "code"
        ? `Write, debug, review and ${state.config.code_execution ? "run" : "explain"} code, with live docs, package versions and GitHub.`
        : `Live answers from ${state.config.tools.length - 3} sources: web, news, X & Reddit, markets, sports, weather and more.`;
    }
  }

  // ---------------------------------------------------------------- composer
  function autoresize() {
    ui.input.style.height = "auto";
    ui.input.style.height = Math.min(ui.input.scrollHeight, 260) + "px";
    if (!state.controller) ui.send.disabled = !ui.input.value.trim() && !state.attachments.some((a) => a.status === "ready");
  }

  // ---------------------------------------------------------------- wire up
  function bind() {
    ui.form.onsubmit = (e) => { e.preventDefault(); state.controller ? stopStreaming() : sendMessage(ui.input.value); };
    ui.input.addEventListener("input", autoresize);
    ui.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!state.controller) sendMessage(ui.input.value); }
    });
    // attachments: button, drag & drop anywhere, paste screenshots
    $("#attachBtn").onclick = () => $("#fileInput").click();
    $("#fileInput").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
    let dragDepth = 0;
    window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; document.body.classList.add("dragging"); } });
    window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault(); dragDepth = 0; document.body.classList.remove("dragging");
      if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]);
    });
    ui.input.addEventListener("paste", (e) => {
      const files = [...(e.clipboardData?.files || [])];
      if (files.length) { e.preventDefault(); addFiles(files); }
    });
    bindAuth();
    document.querySelectorAll(".mode-btn").forEach((b) => (b.onclick = () => { setMode(b.dataset.mode); ui.input.focus(); }));
    $("#newChat").onclick = newChat;
    $("#openSidebar").onclick = () => setSidebar(true);
    $("#closeSidebar").onclick = () => setSidebar(false);
    ui.scrim.onclick = () => setSidebar(false);
    ui.search.oninput = renderHistory;
    $("#openSettings").onclick = () => openSettings();
    $("#settingsForm").addEventListener("submit", (e) => { if (e.submitter?.value === "save") saveSettings(); });
    $("#toggleAllTools").onclick = () => {
      const boxes = [...ui.toolGrid.querySelectorAll("input")];
      const all = boxes.every((b) => b.checked);
      boxes.forEach((b) => (b.checked = !all));
    };
    ui.modelBtn.onclick = (e) => { e.stopPropagation(); ui.modelMenu.hidden = !ui.modelMenu.hidden; };
    document.addEventListener("click", (e) => { if (!e.target.closest(".model-picker")) ui.modelMenu.hidden = true; });
    $("#themeToggle").onclick = () => {
      const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("pulse.theme", next); } catch { /* ignore */ }
    };
    ui.thread.addEventListener("scroll", () => { ui.scrollDown.hidden = nearBottom() || ui.welcome.hidden === false; });
    ui.scrollDown.onclick = () => scrollToBottom(true);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
      if (e.key === "Escape") {
        if (!ui.modelMenu.hidden) ui.modelMenu.hidden = true;
        else if (state.controller) stopStreaming();
      }
    });
  }

  function renderStarters() {
    ui.starters.innerHTML = "";
    for (const s of STARTERS[state.settings.mode] || STARTERS.chat) {
      const b = el("button", "starter", `<span class="ico">${s.ico}</span><span class="t">${esc(s.t)}</span><span class="d">${esc(s.d)}</span>`);
      b.onclick = () => sendMessage(s.p);
      ui.starters.append(b);
    }
  }

  async function init() {
    bind(); setMode(state.settings.mode);
    try { state.config = await api("/api/config"); }
    catch (e) { toast("Server not reachable: " + e.message, 6000); return; }

    // older versions saved a bare Groq model id
    if (state.settings.model && !state.settings.model.includes(":")) state.settings.model = "groq:" + state.settings.model;
    // a saved model may have been retired, or its key removed
    if (!modelAvailable(state.settings.model)) state.settings.model = state.config.default_model;
    setModel(state.settings.model);
    renderUser();

    const badge = $("#storageBadge");
    badge.className = "storage " + (state.config.storage_warning ? "warn" : "ok");
    badge.lastElementChild.textContent = state.config.storage === "supabase" ? "Synced to Supabase" : "Saved locally (SQLite)";
    if (state.config.storage_warning) badge.title = state.config.storage_warning;

    let signedIn = true;
    try { signedIn = await initAuth(); }
    catch (e) { toast("Login unavailable: " + e.message, 6000); signedIn = false; }
    if (!signedIn) return; // the login screen calls loadHistory() after sign-in

    await loadHistory();
    const linked = (location.hash.match(/^#c=([\w-]+)$/) || [])[1];
    if (linked) openConversation(linked);
    if (!providers().some(hasKey)) openSettings("groq");
    ui.input.focus();
  }

  init();
})();
