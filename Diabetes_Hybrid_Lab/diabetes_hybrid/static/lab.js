const demoMessages = JSON.parse(document.querySelector("#client-messages").textContent);
function translateDemo(message) {
  return demoMessages[message] ?? message;
}

const form = document.querySelector("#demo-form");
const question = document.querySelector("#question");
const chatHistory = document.querySelector("#chat-history");
const clearButton = document.querySelector("#clear-chat");
const intro = document.querySelector("#chat-intro");
const agentNames = document.documentElement.lang === "vi" ? {
  Coordinator: "Điều phối", "Data Agent": "Xác thực dữ liệu", "Rule Agent": "Kiểm tra quy tắc",
  "DL Prediction Agent": "Mô hình dự đoán", "KG/RAG Agent": "Truy xuất tri thức",
  "Explanation Agent": "Giải thích", "Monitoring Agent": "Giám sát",
} : {};
let context = null;

for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => {
    question.value = button.dataset.prompt;
    question.focus();
  });
}

clearButton.addEventListener("click", () => {
  context = null;
  chatHistory.replaceChildren();
  clearButton.hidden = true;
  if (intro) intro.hidden = false;
  question.value = "";
  question.focus();
});

function add(parent, tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  parent.append(node);
  return node;
}

function sourceLink(parent, title, url) {
  const target = new URL(url);
  if (target.protocol !== "https:") return;
  const link = add(parent, "a", title);
  link.href = target.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
}

function showReply(turn, result, output) {
  const reply = result.knowledge;
  output.textContent = reply.answer || reply.message;
  if (result.report.review.status !== "none") {
    add(turn, "p", result.report.review.status === "urgent" ? translateDemo("Cần ưu tiên cấp cứu")
      : translateDemo("Cần người có chuyên môn xem xét"), "notice");
  }
  if (reply.next_step) {
    add(turn, "h4", translateDemo("Bước tiếp theo"));
    add(turn, "p", reply.next_step);
  }
  if (reply.support_url) sourceLink(turn, translateDemo("Hướng dẫn cấp cứu từ CDC ↗"), reply.support_url);
  if (reply.evidence?.length) {
    const details = add(turn, "details", "", "chat-sources");
    details.open = true;
    add(details, "summary", translateDemo("Nguồn và bằng chứng"));
    for (const item of reply.evidence) {
      const evidence = add(details, "article", "", "evidence");
      add(evidence, "strong", item.title);
      if (reply.status === "ok" && item.text !== reply.answer) add(evidence, "p", item.text);
      const tier = item.source.tier === "Supplementary" ? translateDemo(" · Nguồn bổ sung") : "";
      add(evidence, "div", "");
      sourceLink(evidence, item.source.publisher + " · " + item.source.title + tier + " ↗", item.source.url);
    }
  }
  const trace = add(turn, "details", "");
  add(trace, "summary", translateDemo("Các bước xử lý"));
  for (const event of result.trace) {
    add(trace, "div", (agentNames[event.agent] || event.agent) + " · " + event.payload.summary);
  }
}

form.addEventListener("submit", async event => {
  if (event.submitter?.value !== "qa") return;
  event.preventDefault();
  const text = question.value.trim();
  if (!text) { question.focus(); return; }
  if (intro) intro.hidden = true;
  for (const previous of chatHistory.querySelectorAll(".chat-sources[open]")) {
    previous.open = false;
  }
  const turn = add(chatHistory, "article", "", "chat-turn");
  add(turn, "h4", translateDemo("Câu hỏi"));
  add(turn, "p", text, "chat-question");
  add(turn, "h4", translateDemo("Trả lời"));
  const output = add(turn, "p", translateDemo("Đang tra cứu…"));
  event.submitter.disabled = true;
  clearButton.disabled = true;
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: "qa", question: text, context}),
    });
    if (!response.ok) throw new Error("Request failed");
    const result = await response.json();
    showReply(turn, result, output);
    context = result.knowledge.status === "ok" ? result.knowledge.context : null;
    question.value = "";
  } catch {
    context = null;
    output.textContent = translateDemo("Chưa kết nối được trợ lý. Hãy thử lại.");
  } finally {
    event.submitter.disabled = false;
    clearButton.disabled = false;
    clearButton.hidden = false;
    turn.scrollIntoView({block: "nearest"});
  }
});
