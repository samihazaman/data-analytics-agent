const API = "";

const messagesEl  = document.getElementById("messages");
const form        = document.getElementById("chat-form");
const input       = document.getElementById("question-input");
const sendBtn     = document.getElementById("send-btn");
const tableList   = document.getElementById("table-list");
const fileInput   = document.getElementById("file-input");
const uploadArea  = document.getElementById("upload-area");
const uploadLabel = document.getElementById("upload-label");
const uploadStatus = document.getElementById("upload-status");
const resetBtn    = document.getElementById("reset-btn");

// ── Table list ────────────────────────────────────────────────

async function refreshTables() {
  try {
    const res = await fetch(`${API}/api/tables`);
    const { tables } = await res.json();
    tableList.innerHTML = "";
    if (!tables.length) {
      tableList.innerHTML = '<li class="empty">No datasets loaded</li>';
      return;
    }
    tables.forEach(({ name, rows, columns }) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <span class="tname">${name}</span>
        <span class="tmeta">${rows.toLocaleString()} rows · ${columns.length} cols</span>
      `;
      tableList.appendChild(li);
    });
  } catch {
    tableList.innerHTML = '<li class="empty">Could not load tables</li>';
  }
}

// ── Chat ──────────────────────────────────────────────────────

function appendMessage(role, html) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `<div class="bubble">${html}</div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function showTyping() {
  const div = document.createElement("div");
  div.className = "message assistant typing";
  div.id = "typing-indicator";
  div.innerHTML = `<div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
  document.getElementById("typing-indicator")?.remove();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswer(text) {
  // Bold **text**
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Italic *text*
  text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");
  // Inline code `code`
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Split into lines and process
  const lines = text.split("\n");
  const output = [];
  let inList = false;

  for (const line of lines) {
    const numbered = line.match(/^\d+\.\s+(.*)/);
    const bullet   = line.match(/^[-•*]\s+(.*)/);

    if (bullet) {
      if (!inList) { output.push("<ul>"); inList = true; }
      output.push(`<li>${bullet[1]}</li>`);
    } else if (numbered) {
      if (!inList) { output.push("<ol>"); inList = true; }
      output.push(`<li>${numbered[1]}</li>`);
    } else {
      if (inList) { output.push(inList === "ol" ? "</ol>" : "</ul>"); inList = false; }
      output.push(line === "" ? "<br>" : `<span>${line}</span><br>`);
    }
  }

  if (inList) output.push("</ul>");
  return output.join("");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  input.value = "";
  autoResize();
  setLoading(true);

  appendMessage("user", escapeHtml(question));
  showTyping();

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    removeTyping();

    if (!res.ok) {
      appendMessage("assistant", `<span style="color:var(--error)">Error: ${escapeHtml(data.detail || "Unknown error")}</span>`);
    } else {
      let html = formatAnswer(escapeHtml(data.answer));
      if (data.images && data.images.length > 0) {
        html += data.images.map(b64 =>
          `<img src="data:image/png;base64,${b64}" class="chart-img" alt="Chart" />`
        ).join("");
      }
      appendMessage("assistant", html);
    }
  } catch (err) {
    removeTyping();
    appendMessage("assistant", `<span style="color:var(--error)">Network error — is the server running?</span>`);
  } finally {
    setLoading(false);
  }
});

function setLoading(on) {
  sendBtn.disabled = on;
  input.disabled = on;
}

// ── Auto-resize textarea ──────────────────────────────────────

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}

input.addEventListener("input", autoResize);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ── File upload ───────────────────────────────────────────────

function setUploadStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `upload-status ${type}`;
}

async function uploadFile(file) {
  if (!file) return;

  const ext = file.name.split(".").pop().toLowerCase();
  if (!["csv", "xlsx", "parquet"].includes(ext)) {
    setUploadStatus("Only .csv, .xlsx, and .parquet files are supported.", "error");
    return;
  }

  setUploadStatus("Uploading...", "loading");
  uploadArea.style.pointerEvents = "none";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      setUploadStatus(data.detail || "Upload failed.", "error");
    } else {
      setUploadStatus(`✓ Loaded "${data.table_name}" — ${data.rows.toLocaleString()} rows`, "success");
      uploadLabel.innerHTML = `<strong>${file.name}</strong><br /><small>Click to replace</small>`;
      await refreshTables();
      appendMessage("assistant", `Dataset <strong>${escapeHtml(data.table_name)}</strong> loaded — ${data.rows.toLocaleString()} rows, ${data.columns.length} columns.<br />Columns: <em>${data.columns.slice(0, 8).join(", ")}${data.columns.length > 8 ? "…" : ""}</em><br /><br />What would you like to know about it?`);
    }
  } catch {
    setUploadStatus("Upload failed — server unreachable.", "error");
  } finally {
    uploadArea.style.pointerEvents = "";
  }
}

fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  uploadFile(e.dataTransfer.files[0]);
});

// ── Reset ─────────────────────────────────────────────────────

resetBtn.addEventListener("click", async () => {
  if (!confirm("Reset the session? This clears conversation memory and removes uploaded datasets.")) return;
  await fetch(`${API}/api/reset`, { method: "POST" });
  messagesEl.innerHTML = "";
  uploadLabel.innerHTML = `Drop a CSV, Excel, or Parquet file<br /><small>or click to browse</small>`;
  setUploadStatus("", "");
  appendMessage("assistant", "Session reset. The Sleep Health dataset is loaded and ready.");
  await refreshTables();
});

// ── Init ──────────────────────────────────────────────────────

refreshTables();
