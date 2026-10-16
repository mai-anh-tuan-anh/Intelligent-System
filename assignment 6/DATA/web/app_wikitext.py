"""WikiText-103 RNN demo. Run: python app.py (from any working directory)."""
from __future__ import annotations

import json
import math
import os
import random
import re
import unicodedata
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
WEIGHTS = MODEL_DIR / "best_rnn_state.pt"
METADATA = MODEL_DIR / "metadata.json"
HOST = "127.0.0.1"
PORT = int(os.environ.get("RNN_WEB_PORT", "8000"))
MAX_BODY = 16_384

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:
    class WordRNN(nn.Module):
        """Exactly the architecture used in the training notebook."""
        def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.rnn = nn.RNN(embed_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
            self.linear = nn.Linear(hidden_dim, vocab_size)

        def forward(self, ids, hidden=None):
            emb = self.embedding(ids)
            states, hidden = self.rnn(emb, hidden)
            return self.linear(states), hidden


def clean_line(line: str) -> str:
    line = unicodedata.normalize("NFC", str(line)).strip().lower()
    if not line or re.fullmatch(r"=+\s*[^=]+\s*=+", line):
        return ""
    return re.sub(r"\s+", " ", line)


def tokenize(line: str) -> list[str]:
    return clean_line(line).split()


class Predictor:
    def __init__(self):
        self.error = None
        self.meta = None
        self.model = None
        if torch is None:
            self.error = "Thiếu PyTorch trong môi trường Python hiện tại. Chạy: python -m pip install torch"
            return
        if not WEIGHTS.is_file() or not METADATA.is_file():
            self.error = (
                "Chưa thấy models/best_rnn_state.pt và models/metadata.json. "
                "Hãy chạy notebook đến cell cuối để xuất cả hai file."
            )
            return
        try:
            self.meta = json.loads(METADATA.read_text(encoding="utf-8"))
            if self.meta.get("tokenizer") != "NFC_lower_whitespace_v1":
                raise ValueError("Phiên bản tokenizer trong metadata không khớp với web")
            self.itos = self.meta["vocab"]
            self.stoi = {word: i for i, word in enumerate(self.itos)}
            if self.itos[:2] != ["<pad>", "<unk>"]:
                raise ValueError("Hai token đầu phải là <pad>, <unk>")
            self.seq_len = int(self.meta["seq_len"])
            self.model = WordRNN(
                len(self.itos), int(self.meta["embed_dim"]), int(self.meta["hidden_dim"])
            )
            state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
        except Exception as exc:
            self.model = None
            self.error = f"Không nạp được mô hình: {exc}"

    def logits(self, tokens: list[str]):
        ids = [self.stoi.get(token, 1) for token in tokens[-self.seq_len:]]
        tensor = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            output, _ = self.model(tensor)
            return output[0, -1].float().cpu()

    def predict(self, prompt: str, top_k: int):
        tokens = tokenize(prompt)
        if not tokens:
            raise ValueError("Hãy nhập ít nhất một từ tiếng Anh.")
        used = tokens[-self.seq_len:]
        logits = self.logits(used)
        # Build a fresh ordinary tensor outside inference_mode for masking.
        logits = logits.clone()
        logits[0] = -float("inf")
        probabilities = torch.softmax(logits, dim=0)
        unknown_probability = float(probabilities[1])
        # Keep the original model probabilities. Only exclude the aggregate
        # <unk> label from the list of words that a person can actually use.
        known_probabilities = probabilities.clone()
        known_probabilities[:2] = -1.0
        values, indices = torch.topk(known_probabilities, min(top_k, len(self.itos) - 2))
        items = [
            {"word": self.itos[int(i)], "probability": float(v), "rank": rank}
            for rank, (i, v) in enumerate(zip(indices, values), start=1)
        ]
        mass = float(values.sum())
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * positive.log()).sum())
        top1 = items[0]["probability"]
        top2 = items[1]["probability"] if len(items) > 1 else float(torch.topk(known_probabilities, min(2, len(self.itos) - 2)).values[-1])
        return {
            "context": used,
            "truncated": len(tokens) > self.seq_len,
            "oov": [token for token in used if token not in self.stoi],
            "top_k": items,
            "top1_probability": items[0]["probability"],
            "confidence_proxy": top1,
            "top1_top2_margin": max(0.0, top1 - top2),
            "effective_candidates": math.exp(entropy),
            "top1_surprisal_bits": -math.log2(max(top1, 1e-30)),
            "top_k_mass": mass,
            "other_mass": max(0.0, 1.0 - mass),
            "unknown_probability": unknown_probability,
            "model_top1_word": self.itos[int(probabilities.argmax())],
            "entropy_nats": entropy,
            "entropy_normalized": entropy / math.log(len(self.itos) - 1),
            "oov_rate": len([t for t in used if t not in self.stoi]) / len(used),
            "vocab_size": len(self.itos),
            "context_limit": self.seq_len,
            "note": "Top-k chỉ gồm token có thể hiển thị. Xác suất gốc không chuẩn hóa lại: phần còn lại bao gồm cả <unk>. Đây không phải độ tin cậy đã hiệu chuẩn.",
        }

    def generate(self, prompt: str, steps: int, temperature: float, sample_top_k: int, seed: int):
        tokens = tokenize(prompt)
        if not tokens:
            raise ValueError("Hãy nhập ít nhất một từ tiếng Anh.")
        rng = random.Random(seed)
        trace = []
        for step in range(steps):
            raw_logits = self.logits(tokens).clone()
            raw_logits[0] = -float("inf")
            raw_probabilities = torch.softmax(raw_logits, dim=0)
            raw_positive = raw_probabilities[raw_probabilities > 0]
            raw_entropy = float(-(raw_positive * raw_positive.log()).sum())
            logits = raw_logits.clone() / temperature
            logits[:2] = -float("inf")
            values, indices = torch.topk(logits, min(sample_top_k, len(self.itos) - 2))
            sampling_probs = torch.softmax(values, dim=0).tolist()
            index = rng.choices(range(len(indices)), weights=sampling_probs, k=1)[0]
            word = self.itos[int(indices[index])]
            trace.append({
                "step": step + 1,
                "word": word,
                "sampling_probability": sampling_probs[index],
                "model_probability": float(raw_probabilities[int(indices[index])]),
                "model_entropy_nats": raw_entropy,
                "options": [
                    {"word": self.itos[int(i)], "probability": float(p)}
                    for i, p in zip(indices[:5], sampling_probs[:5])
                ],
            })
            tokens.append(word)
        return {
            "text": " ".join(tokens),
            "prompt_tokens": len(tokenize(prompt)),
            "generated_tokens": trace,
            "temperature": temperature,
            "sample_top_k": sample_top_k,
            "unique_token_ratio": len({entry["word"] for entry in trace}) / len(trace),
            "note": "sampling_probability được chuẩn hóa sau temperature và top-k sampling.",
        }


PREDICTOR = Predictor()

PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RNN Studio · WikiText-103</title>
<style>
:root{--bg:#f4f6fb;--ink:#152039;--muted:#62708c;--panel:#fff;--line:#dfe5f0;--blue:#395ee6;--teal:#15a69a;--red:#bd4850}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}
header{background:#17254b;color:#fff;padding:24px max(20px,calc((100vw - 1200px)/2));display:flex;align-items:center;gap:18px;justify-content:space-between}
header strong{font-size:21px;letter-spacing:.02em}header small{display:block;color:#c6d4f4;font-size:12px}header .badge{background:#2d4176;padding:6px 12px;border-radius:99px;font-size:12px}
main{max-width:1200px;margin:27px auto;padding:0 20px 45px}.intro{margin:0 0 20px;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(290px,380px) minmax(0,1fr);gap:18px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 5px 22px #2638600a;padding:23px}.panel h2{margin:0 0 6px;font-size:19px}.small{color:var(--muted);font-size:13px}
label{display:block;font-weight:650;font-size:13px;margin:17px 0 6px}textarea,input,select{font:inherit;border:1px solid #cbd5e5;border-radius:9px;background:white;color:var(--ink);outline:none}textarea:focus,input:focus{border-color:var(--blue);box-shadow:0 0 0 3px #395ee621}textarea{width:100%;min-height:120px;padding:12px;resize:vertical}input[type=number]{width:100%;padding:9px 10px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.range{display:flex;align-items:center;gap:10px}.range input{width:100%;accent-color:var(--blue)}button{border:0;background:var(--blue);color:white;font-weight:700;border-radius:9px;padding:11px 17px;cursor:pointer}button:hover{filter:brightness(.95)}button:disabled{opacity:.5;cursor:wait}.actions{display:flex;gap:10px;margin-top:20px}.secondary{background:#edf1ff;color:#304ba8}.msg{padding:11px 13px;border-radius:9px;background:#fff3f2;color:#922e38;margin:0 0 16px}.ok{background:#e8f7f2;color:#25695d}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin:13px 0 20px}.stat{border:1px solid var(--line);background:#f9fbff;border-radius:11px;padding:11px}.stat b{display:block;font-size:20px}.stat small{font-size:11px;color:var(--muted)}
.barrow{display:grid;grid-template-columns:100px minmax(0,1fr) 58px;align-items:center;gap:10px;margin:10px 0;font-size:13px}.barrow .word{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:650}.bar{height:20px;border-radius:6px;background:#eef2fc;overflow:hidden}.fill{height:100%;min-width:2px;background:linear-gradient(90deg,#6580ef,#365be4);border-radius:6px}.barrow .pct{text-align:right;font-variant-numeric:tabular-nums}
.context{display:flex;gap:5px;flex-wrap:wrap;margin:12px 0}.chip{background:#ecf2ff;color:#304878;padding:3px 8px;border-radius:5px;font-size:12px}.chip.oov{background:#fff0db;color:#985c13}.generated{padding:15px;background:#f7f9ff;border-radius:10px;line-height:2;word-wrap:break-word}.token{display:inline-block;padding:2px 5px;margin:2px;border-bottom:2px solid #8b9ef2;cursor:default}.token:hover{background:#e2e9ff}
.trace{max-height:310px;overflow:auto;border:1px solid var(--line);border-radius:9px;margin-top:13px}.trace table{width:100%;border-collapse:collapse;font-size:13px}.trace td,.trace th{padding:7px 10px;border-bottom:1px solid #edf0f5;text-align:left}.trace th{position:sticky;top:0;background:#f4f7fc}.warning{border-left:3px solid #e5a642;padding-left:12px;color:var(--muted);font-size:13px;margin-top:20px}
footer{color:var(--muted);font-size:12px;margin-top:22px}details{margin-top:18px;color:var(--muted);font-size:13px}summary{cursor:pointer;color:var(--ink);font-weight:650}
.vizgrid{display:grid;grid-template-columns:200px minmax(0,1fr);gap:18px;align-items:center;margin:16px 0}.donut{width:170px;height:170px;border-radius:50%;display:grid;place-items:center;margin:auto}.donut-inner{width:112px;height:112px;border-radius:50%;background:white;display:grid;place-items:center;text-align:center;font-weight:750;line-height:1.25}.donut-inner small{font-size:11px;font-weight:400;color:var(--muted)}.metriclist{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:12px 0}.metric{background:#f8faff;border:1px solid var(--line);border-radius:9px;padding:10px}.metric strong{display:block;font-size:16px}.metric span{font-size:11px;color:var(--muted)}.chartgrid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.chartbox{border:1px solid var(--line);border-radius:11px;padding:12px}.chartbox h3{font-size:14px;margin:0 0 8px}.linechart{width:100%;height:170px}.legend{font-size:11px;color:var(--muted)}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 12px}.heatmap{overflow-x:auto}.heatmap table{border-collapse:separate;border-spacing:4px;font-size:11px}.heatmap th{text-align:center;font-weight:600;min-width:42px}.heatmap td{width:48px;height:35px;border-radius:5px;text-align:center;font-variant-numeric:tabular-nums}.heatmap .rowname{text-align:left;min-width:74px;background:none!important}.report{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.report span{border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;color:var(--muted)}
@media(max-width:860px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}header{padding:20px}}
@media(max-width:600px){.vizgrid,.chartgrid{grid-template-columns:1fr}.metriclist{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><div><strong>RNN Studio</strong><small>Word-level language model · WikiText-103</small></div><span class="badge">SimpleRNN · PyTorch</span></header>
<main><p class="intro">Khám phá cách mô hình chọn từ tiếp theo từ ngữ cảnh tiếng Anh và tạo chuỗi từng bước.</p>
<div id="notice"></div><div class="grid">
<section class="panel"><h2>Nhập ngữ cảnh</h2><div class="small">Mô hình huấn luyện bằng dữ liệu WikiText-103 tiếng Anh. Có thể nhập một phần câu.</div>
<label for="prompt">Văn bản tiếng Anh</label><textarea id="prompt" placeholder="the history of the city">the history of the city</textarea>
<div class="row"><div><label for="topk">Số gợi ý</label><input id="topk" type="number" min="1" max="30" value="10"></div><div><label for="steps">Số token sinh</label><input id="steps" type="number" min="1" max="100" value="25"></div></div>
<label for="temp">Temperature <span id="tempValue">1.0</span></label><div class="range"><span>0.2</span><input id="temp" type="range" min="0.2" max="2.0" step="0.1" value="1.0"><span>2.0</span></div>
<div class="row"><div><label for="samplek">Sampling top-k</label><input id="samplek" type="number" min="1" max="100" value="30"></div><div><label for="seed">Seed</label><input id="seed" type="number" min="0" max="999999" value="42"></div></div>
<div class="actions"><button id="predict">Dự đoán từ tiếp</button><button id="generate" class="secondary">Sinh văn bản</button></div>
<div class="warning">Temperature chỉ tác động đến việc lấy mẫu khi sinh văn bản. Xác suất ở biểu đồ dự đoán luôn tính tại temperature = 1.</div></section>
<section class="panel"><h2>Phân tích dự đoán</h2><div class="small" id="subtitle">Nhấn “Dự đoán từ tiếp” để xem phân phối.</div><div id="predictResult"></div></section></div>
<section class="panel" style="margin-top:18px"><h2>Văn bản được sinh</h2><div class="small">Di chuột lên token để xem xác suất lấy mẫu ở từng bước.</div><div id="generateResult" style="margin-top:14px" class="small">Chọn “Sinh văn bản” để xem kết quả và lịch sử các bước.</div></section>
<section class="panel" style="margin-top:18px"><h2>Hiệu năng mô hình</h2><div class="small">Chỉ số từ checkpoint và tập test, cùng các đường học qua từng epoch.</div><div id="modelMetrics" class="small" style="margin-top:12px">Đang đọc metadata...</div></section>
<details><summary>Chỉ số này có ý nghĩa gì?</summary><p><b>Top-1 probability:</b> xác suất mô hình gán cho token được xếp đầu, trên từ vựng rút gọn. <b>Entropy:</b> mức phân tán của phân phối; càng cao thì mô hình càng ít tập trung vào một số token. <b>Top-k mass:</b> tổng xác suất của k gợi ý đang hiển thị. <b>OOV:</b> token nhập vào không có trong từ vựng, được thay bằng &lt;unk&gt;. <b>P(&lt;unk&gt;):</b> xác suất mô hình dành cho toàn bộ nhóm từ ngoài từ vựng, không phải một từ cụ thể. Các giá trị này không phải độ chính xác hay confidence đã hiệu chuẩn.</p></details>
<footer>Dữ liệu: WikiText-103 (tiếng Anh). Nội dung sinh có thể lặp hoặc thiếu mạch lạc do giới hạn của SimpleRNN và tập huấn luyện được chọn.</footer></main>
<script>
const $=id=>document.getElementById(id),escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(x*100).toFixed(x<.01?2:1)+'%';
function notice(message,ok=false){$('notice').innerHTML=message?'<div class="msg '+(ok?'ok':'')+'">'+escapeHTML(message)+'</div>':'';}
async function request(path,payload){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let body;try{body=await response.json()}catch{throw Error('Máy chủ trả phản hồi không hợp lệ.')};if(!response.ok)throw Error(body.error||'Yêu cầu thất bại');return body;}
async function run(button,path,payload,render){button.disabled=true;notice('');try{render(await request(path,payload))}catch(error){notice(error.message)}finally{button.disabled=false}}
function lineChart(rows,first,second,asPercent=false){
 if(!rows.length)return '<p class="small">Checkpoint chưa có lịch sử huấn luyện.</p>';
 const width=480,height=160,pad=20;
 const values=rows.flatMap(r=>[Number(r[first]),Number(r[second])]).filter(Number.isFinite);
 let lo=Math.min(...values),hi=Math.max(...values);if(hi-lo<.0001){hi=lo+1}
 const xy=(i,y)=>`${pad+i*(width-2*pad)/Math.max(1,rows.length-1)},${height-pad-(y-lo)*(height-2*pad)/(hi-lo)}`;
 const poly=key=>rows.map((r,i)=>xy(i,Number(r[key]))).join(' ');
 const fmt=n=>asPercent?pct(n):Number(n).toFixed(3);
 return `<svg class="linechart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Đường train và validation"><path d="M${pad},${height-pad} H${width-pad}" stroke="#cbd5e1"/><polyline points="${poly(first)}" fill="none" stroke="#395ee6" stroke-width="3"/><polyline points="${poly(second)}" fill="none" stroke="#15a69a" stroke-width="3"/><text x="${pad}" y="15" fill="#62708c" font-size="12">${fmt(hi)}</text><text x="${pad}" y="${height-1}" fill="#62708c" font-size="12">${fmt(lo)}</text></svg><div class="legend"><i style="background:#395ee6"></i>Train<i style="background:#15a69a"></i>Validation · ${rows.length} epoch</div>`;
}
function modelMetrics(s){
 const m=s.metrics||{},h=s.history||[],metric=(name,value)=>`<div class="stat"><b>${escapeHTML(value)}</b><small>${escapeHTML(name)}</small></div>`;
 const view=[['Test loss',Number.isFinite(m.loss)?m.loss.toFixed(3):'—'],['Test perplexity',Number.isFinite(m.perplexity)?m.perplexity.toFixed(1):'—'],['Test top-1',Number.isFinite(m.top1)?pct(m.top1):'—'],['Test top-5',Number.isFinite(m.top5)?pct(m.top5):'—']].map(x=>metric(...x)).join('');
 const budget=s.subset_token_budget||{};
 const sample=budget.train==null?'Toàn bộ train':`Subset train ${Number(budget.train).toLocaleString('en-US')} token`;
 const report=[sample,`Vocab ${Number(s.vocab_size).toLocaleString('en-US')}`,`Context ${s.seq_len} token`,`Embedding ${s.embed_dim}`,`Hidden ${s.hidden_dim}`,`Train ${Number(s.train_windows).toLocaleString('en-US')} windows`,`Validation ${Number(s.validation_windows).toLocaleString('en-US')} windows`,`Test ${Number(s.test_windows).toLocaleString('en-US')} windows`].map(x=>`<span>${escapeHTML(x)}</span>`).join('');
 $('modelMetrics').innerHTML=`<div class="cards">${view}</div><div class="report">${report}</div><div class="chartgrid"><div class="chartbox"><h3>Cross entropy theo epoch</h3>${lineChart(h,'train_loss','val_loss')}</div><div class="chartbox"><h3>Top-1 accuracy theo epoch</h3>${lineChart(h,'train_top1','val_top1',true)}</div></div><p class="small">Top-1/Top-5 và perplexity đo trên tập test đã dùng trong notebook. Với subset, không nên coi đây là kết quả benchmark WikiText-103 đầy đủ.</p>`;
}
function prediction(data){
 $('subtitle').textContent='Ngữ cảnh đã dùng: '+data.context.length+'/'+data.context_limit+' token'+(data.truncated?' · đã cắt phần đầu':'');
 const cards=[['Top-1 probability',pct(data.top1_probability)],['Entropy',data.entropy_nats.toFixed(2)+' nats'],['Top-k mass',pct(data.top_k_mass)],['P(<unk>)',pct(data.unknown_probability)]].map(([label,value])=>`<div class="stat"><b>${escapeHTML(value)}</b><small>${escapeHTML(label)}</small></div>`).join('');
 const chips=data.context.map(w=>`<span class="chip ${data.oov.includes(w)?'oov':''}">${escapeHTML(w)}</span>`).join('');
 const max=Math.max(...data.top_k.map(x=>x.probability),.00001);
 const bars=data.top_k.map(x=>`<div class="barrow"><span class="word" title="${escapeHTML(x.word)}">${escapeHTML(x.word)}</span><span class="bar"><span class="fill" style="display:block;width:${Math.max(1,x.probability/max*100)}%"></span></span><span class="pct">${pct(x.probability)}</span></div>`).join('');
 const colors=['#395ee6','#15a69a','#e8a14a','#995bda','#e56e89'];let at=0;
 const parts=data.top_k.slice(0,5).map((item,i)=>{const from=at;at+=item.probability*100;return `${colors[i]} ${from.toFixed(3)}% ${at.toFixed(3)}%`});
 const donut=`<div class="donut" style="background:conic-gradient(${parts.join(',')},#e9edf5 ${at.toFixed(3)}% 100%)"><div class="donut-inner">${pct(data.top_k_mass)}<small>Top-k mass</small></div></div>`;
 const secondary=[['P top-1 hiển thị',pct(data.confidence_proxy)],['Top-1 vs top-2',pct(data.top1_top2_margin)],['Số lựa chọn hiệu dụng',data.effective_candidates.toFixed(0)],['Surprisal top-1',data.top1_surprisal_bits.toFixed(2)+' bits'],['Entropy chuẩn hóa',pct(data.entropy_normalized)],['Tỷ lệ OOV đầu vào',pct(data.oov_rate)]].map(([name,value])=>`<div class="metric"><strong>${escapeHTML(value)}</strong><span>${escapeHTML(name)}</span></div>`).join('');
 $('predictResult').innerHTML=`<div class="cards">${cards}</div><div class="small">Token ngữ cảnh</div><div class="context">${chips}</div><div class="vizgrid">${donut}<div class="metriclist">${secondary}</div></div><div class="small">Xếp hạng ${data.top_k.length} token có thể hiển thị · P(&lt;unk&gt;) = ${pct(data.unknown_probability)} · phần ngoài top-k: ${pct(data.other_mass)}</div>${bars}<div class="warning">${escapeHTML(data.note)} P top-1 hiển thị là xác suất token có thể hiển thị cao nhất; nếu P(&lt;unk&gt;) lớn hơn, nhãn gốc được mô hình xếp đầu là &lt;unk&gt;. Chỉ số này chưa được hiệu chuẩn.</div>`;
}
function generated(data){
 const words=data.generated_tokens.map(t=>`<span class="token" title="Bước ${t.step}: P mô hình ${pct(t.model_probability)}; P lấy mẫu ${pct(t.sampling_probability)}">${escapeHTML(t.word)}</span>`).join('');
 const table=data.generated_tokens.map(t=>`<tr><td>${t.step}</td><td>${escapeHTML(t.word)}</td><td>${pct(t.model_probability)}</td><td>${pct(t.sampling_probability)}</td><td>${escapeHTML(t.options.map(o=>o.word).join(', '))}</td></tr>`).join('');
 const points=data.generated_tokens, w=720,h=190,p=22;
 const coords=(key,max)=>points.map((t,i)=>`${p+i*(w-2*p)/Math.max(1,points.length-1)},${h-p-Math.min(1,t[key]/max)*(h-2*p)}`).join(' ');
 const maxEntropy=Math.max(1,...points.map(t=>t.model_entropy_nats));
 const svg=`<svg class="linechart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Diễn biến xác suất và entropy"><path d="M${p},${h-p} H${w-p}" stroke="#cbd5e1"/><polyline points="${coords('model_probability',1)}" stroke="#395ee6" fill="none" stroke-width="3"/><polyline points="${coords('sampling_probability',1)}" stroke="#15a69a" fill="none" stroke-width="3"/><polyline points="${coords('model_entropy_nats',maxEntropy)}" stroke="#e8a14a" fill="none" stroke-width="2"/></svg><div class="legend"><i style="background:#395ee6"></i>P mô hình<i style="background:#15a69a"></i>P lấy mẫu<i style="background:#e8a14a"></i>Entropy (thang riêng)</div>`;
 const heads=points.slice(0,30),topwords=[];
 for(const t of heads)for(const o of t.options)if(!topwords.includes(o.word))topwords.push(o.word);
 const wordRows=topwords.slice(0,8).map(word=>`<tr><th class="rowname">${escapeHTML(word)}</th>${heads.map(t=>{const v=t.options.find(x=>x.word===word)?.probability||0;return `<td title="Bước ${t.step}: ${word} ${pct(v)}" style="background:rgba(57,94,230,${(.08+v*.9).toFixed(3)})">${v>0?pct(v):'·'}</td>`}).join('')}</tr>`).join('');
 const heat=`<div class="heatmap"><table><thead><tr><th>Token / bước</th>${heads.map(t=>`<th>${t.step}</th>`).join('')}</tr></thead><tbody>${wordRows}</tbody></table></div>`;
 $('generateResult').innerHTML=`<div class="generated"><span>${escapeHTML(data.text.split(' ').slice(0,data.prompt_tokens).join(' '))}</span> ${words}</div><div class="cards" style="margin-top:16px"><div class="stat"><b>${points.length}</b><small>Token sinh</small></div><div class="stat"><b>${data.temperature.toFixed(1)}</b><small>Temperature</small></div><div class="stat"><b>${data.sample_top_k}</b><small>Sampling top-k</small></div><div class="stat"><b>${pct(data.unique_token_ratio)}</b><small>Token khác nhau</small></div></div><h3>Xác suất qua từng bước</h3>${svg}<h3>Heatmap các gợi ý xuất hiện nhiều nhất</h3><div class="small">Tối đa 30 bước đầu, 8 token nổi bật; màu đậm hơn biểu thị xác suất lấy mẫu cao hơn.</div>${heat}<h3>Bảng từng bước</h3><div class="trace"><table><thead><tr><th>Bước</th><th>Token</th><th>P mô hình</th><th>P lấy mẫu</th><th>Gợi ý tại bước đó</th></tr></thead><tbody>${table}</tbody></table></div><p class="small">${escapeHTML(data.note)}</p>`;
}
$('temp').addEventListener('input',()=>{$('tempValue').textContent=Number($('temp').value).toFixed(1)});
$('predict').onclick=()=>run($('predict'),'/api/predict',{text:$('prompt').value,top_k:Number($('topk').value)},prediction);
$('generate').onclick=()=>run($('generate'),'/api/generate',{text:$('prompt').value,steps:Number($('steps').value),temperature:Number($('temp').value),sample_top_k:Number($('samplek').value),seed:Number($('seed').value)},generated);
fetch('/api/status').then(r=>r.json()).then(s=>{if(!s.ready)notice(s.error);else{notice('Mô hình đã sẵn sàng · vocabulary '+s.vocab_size+' token',true);modelMetrics(s);$('predict').click()}}).catch(()=>notice('Không kết nối được máy chủ.'));
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def respond(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, status: int, value):
        self.respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.respond(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            status = {"ready": PREDICTOR.model is not None, "error": PREDICTOR.error}
            if PREDICTOR.model is not None:
                status.update({"vocab_size": len(PREDICTOR.itos), "dataset": PREDICTOR.meta.get("dataset"),
                               "metrics": PREDICTOR.meta.get("test_metrics", {}),
                               "history": PREDICTOR.meta.get("history", []),
                               "best_validation_loss": PREDICTOR.meta.get("best_validation_loss"),
                               "subset_token_budget": PREDICTOR.meta.get("subset_token_budget"),
                               "train_windows": PREDICTOR.meta.get("train_windows"),
                               "validation_windows": PREDICTOR.meta.get("validation_windows"),
                               "test_windows": PREDICTOR.meta.get("test_windows"),
                               "seq_len": PREDICTOR.seq_len,
                               "embed_dim": PREDICTOR.meta.get("embed_dim"),
                               "hidden_dim": PREDICTOR.meta.get("hidden_dim")})
            self.json_response(200, status)
        else:
            self.json_response(404, {"error": "Không tìm thấy đường dẫn."})

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/predict", "/api/generate"):
            self.json_response(404, {"error": "Không tìm thấy đường dẫn."})
            return
        if PREDICTOR.model is None:
            self.json_response(503, {"error": PREDICTOR.error})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                raise ValueError("Yêu cầu trống hoặc quá dài (tối đa 16 KB).")
            raw = self.rfile.read(size)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Dữ liệu phải là JSON object.")
            prompt = data.get("text", "")
            if not isinstance(prompt, str) or len(prompt) > 4000:
                raise ValueError("Văn bản cần là chuỗi không quá 4000 ký tự.")
            if path == "/api/predict":
                k = int(data.get("top_k", 10))
                if not 1 <= k <= 30:
                    raise ValueError("top_k phải từ 1 đến 30.")
                result = PREDICTOR.predict(prompt, k)
            else:
                steps = int(data.get("steps", 25))
                temperature = float(data.get("temperature", 1.0))
                sample_top_k = int(data.get("sample_top_k", 30))
                seed = int(data.get("seed", 42))
                if not 1 <= steps <= 100 or not 0.2 <= temperature <= 2.0 or not 1 <= sample_top_k <= 100 or not 0 <= seed <= 999999:
                    raise ValueError("steps: 1–100; temperature: 0.2–2; sample_top_k: 1–100; seed: 0–999999.")
                result = PREDICTOR.generate(prompt, steps, temperature, sample_top_k, seed)
            self.json_response(200, result)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            self.json_response(400, {"error": str(exc)})
        except Exception as exc:
            self.json_response(500, {"error": f"Lỗi suy luận: {exc}"})


if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print(f"Project: {ROOT}")
    print(f"Web: {url}")
    if PREDICTOR.error:
        print(f"Mô hình chưa sẵn sàng: {PREDICTOR.error}")
    else:
        print(f"Đã nạp {len(PREDICTOR.itos)} token từ {METADATA}")
    print("Nhấn Ctrl+C để dừng.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng web.")
    finally:
        server.server_close()
