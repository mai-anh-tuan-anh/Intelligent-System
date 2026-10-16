"""Interactive local web demo for the e-commerce and stock RNN notebook.

Run from this folder with: python app.py
Set RNN_PROJECT_ROOT if the exported notebook artifacts live elsewhere.
"""
from __future__ import annotations

import json
import math
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("RNN_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
PAYLOAD_FILE = ROOT / "outputs" / "metrics" / "web_payload.json"
MODEL_ROOT = ROOT / "models"
WEB_ROOT = Path(__file__).resolve().parent
HTML_FILE = WEB_ROOT / "templates" / "index.html"
STATIC_DIR = WEB_ROOT / "static"
HOST = "127.0.0.1"
PORT = int(os.environ.get("RNN_TIMESERIES_PORT", "8051"))
MAX_BODY = 1_000_000
LOOKBACK_DEFAULT = 14

FALLBACK_DATASETS = {
    "ecommerce": {
        "display_name": "E-commerce service process",
        "features": [
            "delivered_order_count", "avg_approval_hours", "avg_carrier_hours",
            "avg_promised_days", "late_rate", "days_since_previous_service_day",
        ],
        "target_unit": "days",
        "forecast_horizon": "next observed service period",
    },
    "stock": {
        "display_name": "AAPL stock",
        "features": [
            "Open", "High", "Low", "Close", "Volume", "Return_1d",
            "HL_range_pct", "OC_change_pct", "Volume_change_pct",
        ],
        "target_unit": "price units",
        "forecast_horizon": "next trading step",
    },
}
FRAMEWORKS = ("Scratch", "Keras", "PyTorch")

try:
    import numpy as np
except ImportError:
    np = None

try:
    import joblib
except ImportError:
    joblib = None


PAGE = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RNN Forecast Lab</title>
<style>
:root{--bg:#f3f6fc;--ink:#17233b;--muted:#61708a;--line:#dfe6f1;--panel:#fff;--blue:#315be8;--teal:#149a87;--amber:#c5811d;--red:#b94450;--soft:#f7f9fe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{background:linear-gradient(110deg,#15254e,#25458b);color:white;padding:22px max(20px,calc((100vw - 1320px)/2));display:flex;align-items:center;justify-content:space-between;gap:20px}header h1{margin:0;font-size:23px}header p{margin:3px 0 0;color:#d0daf2;font-size:13px}.badge{padding:6px 11px;background:#ffffff20;border:1px solid #ffffff35;border-radius:30px;font-size:12px;white-space:nowrap}
main{max-width:1320px;margin:24px auto;padding:0 18px 45px}.intro{color:var(--muted);margin:0 0 18px}.banner{padding:12px 15px;border-radius:10px;margin:0 0 16px;background:#fff2dd;color:#79500f;border:1px solid #f0d39e}.banner.error{background:#fff0f0;border-color:#f0c6c6;color:#8f3038}.banner.good{background:#e9f8f2;border-color:#bfe8d7;color:#23674f}.hidden{display:none!important}
.topgrid{display:grid;grid-template-columns:minmax(360px,1.2fr) minmax(330px,.8fr);gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:19px;box-shadow:0 5px 20px #1c31500a}.panel h2{font-size:18px;margin:0 0 5px}.sub{color:var(--muted);font-size:12px;margin:0 0 13px}
.toolbar{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field label{display:block;font-size:12px;font-weight:700;margin:0 0 5px}.field select,.field input{width:100%;padding:9px 10px;border:1px solid #cbd5e3;border-radius:8px;background:#fff;color:var(--ink);font:inherit}.tablewrap{overflow:auto;max-height:440px;border:1px solid var(--line);border-radius:10px;margin-top:12px}.inputtable{border-collapse:separate;border-spacing:0;min-width:100%;font-size:12px}.inputtable th,.inputtable td{padding:5px;border-bottom:1px solid #edf0f5;border-right:1px solid #edf0f5;text-align:center;white-space:nowrap}.inputtable th{position:sticky;top:0;background:#f2f5fb;z-index:2;font-weight:700}.inputtable th:first-child{left:0;z-index:3}.inputtable td:first-child{position:sticky;left:0;background:#fafbfe;font-weight:700;z-index:1}.inputtable input{width:105px;padding:6px;border:1px solid #d8dfeb;border-radius:6px;text-align:right;font:12px ui-monospace,Consolas,monospace}.inputtable input:focus{outline:2px solid #aab9fc;border-color:var(--blue)}
.buttons{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px;align-items:center}button,.uploadbtn{border:0;border-radius:8px;padding:9px 13px;background:var(--blue);color:white;font:700 13px system-ui;cursor:pointer;display:inline-flex;align-items:center;gap:6px}button:hover,.uploadbtn:hover{filter:brightness(.96)}button:disabled{opacity:.5;cursor:wait}.secondary{background:#e9efff;color:#2948a7}.light{background:#eef2f8;color:#374762}.uploadbtn input{display:none}.tiny{font-size:11px;color:var(--muted)}
.metriccards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin:12px 0}.metric{border:1px solid var(--line);border-radius:10px;background:#f9fbff;padding:11px}.metric strong{font-size:20px;display:block;overflow-wrap:anywhere}.metric span{font-size:11px;color:var(--muted)}.forecast{border-radius:13px;background:linear-gradient(135deg,#edf2ff,#f7f9ff);padding:17px;margin:12px 0}.forecast .value{font-size:32px;font-weight:800;color:#2449c2;line-height:1.1}.forecast .label{font-size:12px;color:var(--muted);margin-top:6px}.forecast .direction{display:inline-block;margin-top:9px;padding:4px 9px;border-radius:20px;background:#fff;font-weight:700;font-size:12px}.errnote{border-left:3px solid var(--amber);padding:8px 10px;background:#fff9ee;color:#785519;font-size:11px;margin-top:10px}
.compare{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}.compare th,.compare td{text-align:right;padding:8px;border-bottom:1px solid #edf0f5}.compare th:first-child,.compare td:first-child{text-align:left}.compare th{color:var(--muted);font-size:11px}.compare tr.selected{background:#eff3ff;font-weight:700}.sectiongrid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.chartbox{min-width:0}.charthead{display:flex;justify-content:space-between;align-items:center;gap:10px}.charthead select{padding:6px;border:1px solid var(--line);border-radius:7px;background:white;font:12px system-ui}.chart{width:100%;height:250px;display:block}.legend{display:flex;gap:13px;font-size:11px;color:var(--muted);flex-wrap:wrap}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:4px}.heatwrap{overflow:auto}.heat{border-collapse:separate;border-spacing:3px;font-size:10px}.heat th{padding:4px;max-width:100px;white-space:normal}.heat td{min-width:48px;height:31px;border-radius:5px;text-align:center;font:10px ui-monospace,monospace}.heat .rowhead{text-align:left;background:transparent!important;font-weight:700}.footnote{font-size:11px;color:var(--muted);margin-top:10px}details{margin-top:18px}summary{cursor:pointer;font-weight:700}details p{color:var(--muted);font-size:13px}
@media(max-width:900px){.topgrid,.sectiongrid{grid-template-columns:1fr}}@media(max-width:550px){header{align-items:flex-start;flex-direction:column}.toolbar{grid-template-columns:1fr}.panel{padding:14px}.forecast .value{font-size:27px}}
</style>
</head>
<body>
<header><div><h1>RNN Forecast Lab</h1><p>Interactive time-series prediction · E-commerce and AAPL</p></div><span class="badge">Scratch · Keras · PyTorch</span></header>
<main>
<p class="intro">Sửa một cửa sổ gồm 14 bước lịch sử rồi bấm <b>Predict</b>. Web dùng mô hình đã huấn luyện để dự đoán bước tiếp theo; không huấn luyện lại khi bạn bấm nút.</p>
<div id="banner" class="banner hidden"></div>
<div class="topgrid">
 <section class="panel">
  <h2>Đầu vào dự đoán</h2><p class="sub">Các hàng theo thứ tự cũ → mới. Giá trị ban đầu lấy từ cửa sổ dữ liệu cuối trong notebook; bạn có thể chỉnh tay hoặc nạp CSV.</p>
  <div class="toolbar">
   <div class="field"><label for="dataset">Dataset</label><select id="dataset"><option value="ecommerce">E-commerce service</option><option value="stock">AAPL stock</option></select></div>
   <div class="field"><label for="framework">Mô hình</label><select id="framework"><option>Scratch</option><option>Keras</option><option>PyTorch</option></select></div>
  </div>
  <div id="tablewrap" class="tablewrap"><div class="sub" style="padding:12px">Đang tải cấu hình...</div></div>
  <div class="buttons"><button id="predict">Predict</button><button id="reset" class="secondary">Khôi phục cửa sổ mẫu</button><label class="uploadbtn light">Tải CSV<input id="csv" type="file" accept=".csv,text/csv"></label><span class="tiny">CSV: 14 dòng dữ liệu, có hàng tiêu đề tên đặc trưng.</span></div>
  <p class="footnote">Các giá trị trên form là giá trị gốc theo đơn vị của dataset; backend sẽ chuẩn hóa bằng scaler đã lưu trong notebook.</p>
 </section>
 <section class="panel">
  <h2>Kết quả dự đoán</h2><p id="resultSub" class="sub">Chọn dataset, mô hình và nhấn Predict.</p>
  <div id="prediction"><div class="forecast"><div class="value">—</div><div class="label">Chưa có dự đoán mới</div></div></div>
  <h3 style="font-size:14px;margin:17px 0 3px">So sánh trên tập test</h3>
  <div id="comparison" class="sub">Metrics sẽ được đọc từ kết quả notebook.</div>
  <div class="errnote">MAE là sai số tuyệt đối trung bình trên tập test, chỉ dùng làm mốc tham khảo. Đây là bài toán hồi quy nên kết quả không phải xác suất hay khoảng tin cậy được đảm bảo.</div>
 </section>
</div>
<div class="sectiongrid">
 <section class="panel chartbox"><div class="charthead"><div><h2>Lịch sử dự đoán trên tập test</h2><p class="sub">Đường thực tế và dự đoán của mô hình được chọn.</p></div></div><canvas id="historyChart" class="chart"></canvas><div class="legend"><span><i class="dot" style="background:#263c70"></i>Thực tế</span><span><i class="dot" style="background:#315be8"></i>Dự đoán</span></div></section>
 <section class="panel chartbox"><div class="charthead"><div><h2>Đặc trưng đầu vào</h2><p class="sub">Quan sát một đặc trưng qua 14 bước trong cửa sổ hiện tại.</p></div><select id="featureSelect"></select></div><canvas id="inputChart" class="chart"></canvas></section>
 <section class="panel"><h2>Bản đồ đầu vào</h2><p class="sub">Màu đậm/nhạt biểu thị thấp/cao tương đối trong từng đặc trưng của cửa sổ đang nhập.</p><div id="heatmap" class="heatwrap"></div></section>
 <section class="panel chartbox"><div class="charthead"><div><h2>Lịch sử huấn luyện</h2><p class="sub">Loss qua các epoch của mô hình được chọn.</p></div></div><canvas id="lossChart" class="chart"></canvas><div class="legend"><span><i class="dot" style="background:#315be8"></i>Train loss</span><span><i class="dot" style="background:#149a87"></i>Validation loss</span></div></section>
</div>
<details><summary>Demo này xử lý dữ liệu như thế nào?</summary><p>Form nhận đúng LOOKBACK × số đặc trưng. Khi bấm Predict, backend kiểm tra kích thước và giá trị, dùng feature scaler của notebook, chạy mô hình Scratch/Keras/PyTorch, rồi đổi dự đoán về đơn vị gốc bằng target scaler. Chỉ số và biểu đồ lịch sử là kết quả đánh giá đã xuất từ notebook; dự đoán trên form là suy luận mới theo dữ liệu bạn nhập.</p></details>
</main>
<script>
const FALLBACK={
 ecommerce:{display_name:'E-commerce service process',features:['delivered_order_count','avg_approval_hours','avg_carrier_hours','avg_promised_days','late_rate','days_since_previous_service_day'],target_unit:'days',forecast_horizon:'next observed service period'},
 stock:{display_name:'AAPL stock',features:['Open','High','Low','Close','Volume','Return_1d','HL_range_pct','OC_change_pct','Volume_change_pct'],target_unit:'price units',forecast_horizon:'next trading step'}
};
const state={boot:null,ready:false,windows:{},latestResult:null};
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function banner(text,kind=''){const el=$('banner');el.textContent=text;el.className='banner '+kind+(text?'':' hidden')}
function ds(){const k=$('dataset').value;return state.boot?.datasets?.[k]||FALLBACK[k]}
function currentWindow(){const features=ds().features;const rows=[...document.querySelectorAll('#inputBody tr')];return rows.map(r=>features.map((_,j)=>Number(r.querySelector(`[data-col="${j}"]`).value)))}
function renderTable(){const d=ds(),features=d.features||[],w=state.windows[$('dataset').value]||Array.from({length:state.boot?.lookback||14},()=>features.map(()=>0));state.windows[$('dataset').value]=w;let html='<table class="inputtable"><thead><tr><th>Thời điểm</th>'+features.map(f=>`<th title="${esc(f)}">${esc(f)}</th>`).join('')+'</tr></thead><tbody id="inputBody">';for(let i=0;i<w.length;i++){html+=`<tr><td>t-${w.length-1-i||0}</td>`+features.map((f,j)=>`<td><input type="number" step="any" value="${Number(w[i][j]??0)}" data-row="${i}" data-col="${j}" aria-label="${esc(f)} ở bước ${i+1}"></td>`).join('')+'</tr>'}html+='</tbody></table>'; $('tablewrap').innerHTML=html;document.querySelectorAll('#inputBody input').forEach(el=>el.addEventListener('input',refreshInputCharts));renderFeatureOptions();renderComparison();drawHistory();drawLoss();refreshInputCharts()}
function renderFeatureOptions(){const f=ds().features||[],select=$('featureSelect'),previous=select.value;select.innerHTML=f.map((x,i)=>`<option value="${i}">${esc(x)}</option>`).join('');if(previous&&Number(previous)<f.length)select.value=previous;select.onchange=refreshInputCharts}
function fmt(x){if(!Number.isFinite(Number(x)))return '—';return Number(x).toLocaleString('en-US',{maximumFractionDigits:4})}
function getModelSummary(){const d=ds();return d.summary||[]}
function summaryFor(fr){return getModelSummary().find(x=>String(x.framework).toLowerCase()===String(fr).toLowerCase())||{}}
function renderComparison(){const rows=getModelSummary();if(!rows.length){$('comparison').textContent='Chưa có metrics. Hãy chạy cell export cuối notebook và đặt outputs/metrics/web_payload.json vào thư mục outputs/metrics.';return}const names=['Persistence','Scratch','Keras','PyTorch'];$('comparison').innerHTML='<table class="compare"><thead><tr><th>Mô hình</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>'+names.map(n=>{const x=rows.find(r=>r.framework===n);return x?`<tr class="${n===$('framework').value?'selected':''}"><td>${esc(n)}</td><td>${fmt(x.MAE)}</td><td>${fmt(x.RMSE)}</td><td>${fmt(x.R2)}</td></tr>`:''}).join('')+'</tbody></table>'}
function canvasSetup(canvas){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,rect.width*ratio);canvas.height=Math.max(1,rect.height*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);return {ctx,w:rect.width,h:rect.height}}
function drawLines(canvas,sets,labels){const {ctx,w,h}=canvasSetup(canvas);ctx.clearRect(0,0,w,h);const pad={l:48,r:12,t:14,b:28},all=sets.flatMap(s=>s.values).filter(Number.isFinite);if(!all.length){ctx.fillStyle='#71809a';ctx.font='13px system-ui';ctx.fillText('Chưa có dữ liệu biểu đồ',14,28);return}let lo=Math.min(...all),hi=Math.max(...all);if(hi===lo){hi=lo+1}const x=i=>pad.l+i*(w-pad.l-pad.r)/Math.max(sets[0].values.length-1,1),y=v=>h-pad.b-(v-lo)*(h-pad.t-pad.b)/(hi-lo);ctx.strokeStyle='#e5eaf2';ctx.lineWidth=1;for(let k=0;k<4;k++){const yy=pad.t+k*(h-pad.t-pad.b)/3;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillStyle='#748199';ctx.font='10px system-ui';ctx.fillText(fmt(hi-k*(hi-lo)/3),2,yy+3)}for(const s of sets){ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=2;let started=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;if(!started){ctx.moveTo(x(i),y(v));started=true}else ctx.lineTo(x(i),y(v))});ctx.stroke()}ctx.fillStyle='#71809a';ctx.font='10px system-ui';ctx.fillText(labels?.[0]||'đầu chuỗi',pad.l,h-6);ctx.textAlign='right';ctx.fillText(labels?.[1]||'cuối chuỗi',w-pad.r,h-6);ctx.textAlign='left'}
function selectedSeries(){const d=ds(),fr=$('framework').value;return d.series?.[fr]||d.series?.[fr.toLowerCase()]||null}
function drawHistory(){const s=selectedSeries();if(!s){drawLines($('historyChart'),[],[]);return}drawLines($('historyChart'),[{values:(s.actual||[]).map(Number),color:'#263c70'},{values:(s.predicted||[]).map(Number),color:'#315be8'}],[(s.dates||[])[0],(s.dates||[]).at(-1)])}
function drawLoss(){const h=ds().histories?.[$('framework').value]||{};const a=h.train_loss||h.loss||[],b=h.val_loss||h.val_loss||[];drawLines($('lossChart'),[{values:a.map(Number),color:'#315be8'},{values:b.map(Number),color:'#149a87'}],['epoch 1',`epoch ${Math.max(a.length,b.length)}`])}
function refreshInputCharts(){const w=currentWindow();if(!w.length)return;const j=Number($('featureSelect').value||0);drawLines($('inputChart'),[{values:w.map(r=>Number(r[j])),color:'#315be8'}],['bước đầu','bước cuối']);renderHeatmap(w)}
function renderHeatmap(w){const f=ds().features||[];if(!w.length){$('heatmap').innerHTML='';return}let html='<table class="heat"><thead><tr><th></th>'+w.map((_,i)=>`<th>${i+1}</th>`).join('')+'</tr></thead><tbody>';for(let j=0;j<f.length;j++){const vals=w.map(r=>Number(r[j])),lo=Math.min(...vals),hi=Math.max(...vals);html+=`<tr><td class="rowhead">${esc(f[j])}</td>`+vals.map(v=>{const p=hi===lo?.5:(v-lo)/(hi-lo),alpha=.08+p*.36;return `<td style="background:rgba(49,91,232,${alpha})">${fmt(v)}</td>`}).join('')+'</tr>'}html+='</tbody></table>';$('heatmap').innerHTML=html}
function setDatasetDefaults(){const k=$('dataset').value,d=ds();if(state.windows[k])return;const scaled=d.last_window;if(scaled&&state.boot?.raw_windows?.[k])state.windows[k]=state.boot.raw_windows[k].map(r=>r.slice());else state.windows[k]=Array.from({length:state.boot?.lookback||14},()=>d.features.map(()=>0))}
function afterPrediction(r){state.latestResult=r;const selected=$('framework').value;const base=Number(r.latest_observed_target);const predicted=Number(r.prediction);let direction='';if(Number.isFinite(base)){const delta=predicted-base;direction=`<span class="direction">${delta>=0?'Tăng':'Giảm'} so với mốc gần nhất · Δ ${fmt(Math.abs(delta))}</span>`}const mae=Number(r.metrics?.MAE);$('resultSub').textContent=`${r.dataset_name} · ${r.framework} · dự đoán ${r.forecast_horizon}`;$('prediction').innerHTML=`<div class="forecast"><div class="value">${fmt(predicted)} <span style="font-size:14px">${esc(r.target_unit)}</span></div><div class="label">Giá trị dự đoán cho bước tiếp theo</div>${direction}</div><div class="metriccards"><div class="metric"><strong>${fmt(mae)} ${esc(r.target_unit)}</strong><span>MAE trên tập test (mốc sai số tham khảo)</span></div><div class="metric"><strong>${fmt(r.metrics?.RMSE)} ${esc(r.target_unit)}</strong><span>RMSE trên tập test</span></div></div><div class="sub">Input hợp lệ: ${r.lookback} bước × ${r.features.length} đặc trưng · Đã áp dụng scaler của notebook.</div>`;renderComparison()}
async function predict(){const btn=$('predict');btn.disabled=true;banner('Đang chuẩn hóa đầu vào và chạy mô hình…','good');try{const payload={dataset:$('dataset').value,framework:$('framework').value,window:currentWindow()};const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await res.json();if(!res.ok)throw new Error(data.error||'Không dự đoán được');afterPrediction(data);banner('Dự đoán hoàn tất. Đây là inference mới trên dữ liệu bạn vừa nhập.','good')}catch(e){banner(e.message,'error')}finally{btn.disabled=false}}
function parseCSV(text){const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/).filter(Boolean);if(lines.length<2)throw Error('CSV cần có một hàng tiêu đề và các hàng dữ liệu.');const headers=lines[0].split(',').map(x=>x.trim().replace(/^"|"$/g,''));const features=ds().features;const positions=features.map(f=>headers.indexOf(f));if(positions.some(i=>i<0))throw Error('Tên cột CSV phải khớp các đặc trưng: '+features.join(', '));const rows=lines.slice(1).map((line,idx)=>{const parts=line.split(',').map(x=>x.trim().replace(/^"|"$/g,''));return positions.map(p=>{const n=Number(parts[p]);if(!Number.isFinite(n))throw Error(`Dữ liệu CSV không phải số ở dòng ${idx+2}.`);return n})});if(rows.length!==(state.boot?.lookback||14))throw Error(`CSV cần đúng ${state.boot?.lookback||14} dòng dữ liệu, hiện có ${rows.length}.`);state.windows[$('dataset').value]=rows;renderTable();banner('Đã nạp CSV. Kiểm tra thứ tự cũ → mới rồi nhấn Predict.','good')}
async function init(){try{const r=await fetch('/api/bootstrap');const b=await r.json();state.boot=b;state.ready=Boolean(b.ready);if(b.datasets){for(const [k,d] of Object.entries(b.datasets)){if(d.last_window&&b.raw_windows?.[k])state.windows[k]=b.raw_windows[k].map(row=>row.slice())}}if(b.missing?.length)banner('Chưa tìm thấy artifacts đầy đủ. Web đã mở giao diện, nhưng để Predict cần chạy cell xuất cuối notebook và đặt các file models/ cùng outputs/metrics/web_payload.json đúng thư mục. Thiếu: '+b.missing.join(', '),'error');else banner('Đã nạp metrics, cửa sổ mẫu và artifacts mô hình từ notebook.','good')}catch(e){state.boot={lookback:14,datasets:{}};banner('Không đọc được cấu hình notebook: '+e.message,'error')}setDatasetDefaults();renderTable()}
$('dataset').onchange=()=>{setDatasetDefaults();renderTable();state.latestResult=null;$('prediction').innerHTML='<div class="forecast"><div class="value">—</div><div class="label">Nhấn Predict để chạy mô hình</div></div>';};
$('framework').onchange=()=>{renderComparison();drawHistory();drawLoss()};$('predict').onclick=predict;$('reset').onclick=()=>{const k=$('dataset').value;const b=state.boot?.raw_windows?.[k];if(b)state.windows[k]=b.map(r=>r.slice());else state.windows[k]=Array.from({length:state.boot?.lookback||14},()=>ds().features.map(()=>0));renderTable()};$('csv').onchange=async e=>{const file=e.target.files?.[0];if(!file)return;try{parseCSV(await file.text())}catch(err){banner(err.message,'error')}finally{e.target.value=''}};window.addEventListener('resize',()=>{drawHistory();drawLoss();refreshInputCharts()});init();
</script>
</body></html>'''


class NotebookArtifacts:
    def __init__(self):
        self.payload = None
        self.datasets = {}
        self.errors = []
        self.models = {}
        if np is None:
            self.errors.append("Thiếu numpy: cài bằng pip install numpy")
        if joblib is None:
            self.errors.append("Thiếu joblib/scikit-learn: cài bằng pip install joblib scikit-learn")
        if not PAYLOAD_FILE.is_file():
            self.errors.append(f"Không tìm thấy {PAYLOAD_FILE.relative_to(ROOT) if PAYLOAD_FILE.is_relative_to(ROOT) else PAYLOAD_FILE}")
            return
        try:
            self.payload = json.loads(PAYLOAD_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            self.errors.append(f"Không đọc được web_payload.json: {exc}")
            return
        for key, fallback in FALLBACK_DATASETS.items():
            data = self.payload.get(key)
            if not isinstance(data, dict):
                self.errors.append(f"web_payload.json thiếu dataset '{key}'")
                continue
            base = MODEL_ROOT / key
            x_path = base / "feature_scaler.joblib"
            y_path = base / "target_scaler.joblib"
            if not x_path.is_file():
                self.errors.append(str(x_path.relative_to(ROOT)))
            if not y_path.is_file():
                self.errors.append(str(y_path.relative_to(ROOT)))
            x_scaler = y_scaler = None
            if joblib is not None and x_path.is_file() and y_path.is_file():
                try:
                    x_scaler, y_scaler = joblib.load(x_path), joblib.load(y_path)
                except Exception as exc:
                    self.errors.append(f"Không nạp được scaler {key}: {exc}")
            features = data.get("features") or fallback["features"]
            raw_window = None
            if np is not None and x_scaler is not None and data.get("last_window") is not None:
                try:
                    scaled = np.asarray(data["last_window"], dtype=np.float32)
                    raw_window = x_scaler.inverse_transform(scaled).astype(float).tolist()
                except Exception as exc:
                    self.errors.append(f"Không khôi phục được cửa sổ mẫu {key}: {exc}")
            for framework, filename in (("Scratch", "scratch_rnn.npz"), ("Keras", "keras_rnn.keras"), ("PyTorch", "pytorch_rnn.pt")):
                if not (base / filename).is_file():
                    self.errors.append(str((base / filename).relative_to(ROOT)))
            self.datasets[key] = {
                **fallback,
                **data,
                "features": features,
                "lookback": int(self.payload.get("config", {}).get("lookback", LOOKBACK_DEFAULT)),
                "raw_window": raw_window,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
            }

    def bootstrap(self):
        data = {}
        raw_windows = {}
        for key, fallback in FALLBACK_DATASETS.items():
            d = self.datasets.get(key, fallback)
            data[key] = {k: v for k, v in d.items() if k not in ("x_scaler", "y_scaler", "raw_window")}
            if d.get("raw_window") is not None:
                raw_windows[key] = d["raw_window"]
        config = (self.payload or {}).get("config", {})
        return {
            "ready": bool(self.payload) and not self.errors,
            "missing": list(dict.fromkeys(self.errors)),
            "lookback": int(config.get("lookback", LOOKBACK_DEFAULT)),
            "datasets": data,
            "raw_windows": raw_windows,
        }

    def _predict_model(self, dataset_key: str, framework: str, x_scaled):
        cache_key = (dataset_key, framework)
        if cache_key in self.models:
            model = self.models[cache_key]
            return model(x_scaled)
        base = MODEL_ROOT / dataset_key
        n_features = x_scaled.shape[-1]
        hidden_size = int((self.payload or {}).get("config", {}).get("hidden_size", 64))
        x = x_scaled.astype(np.float32)
        if framework == "Scratch":
            path = base / "scratch_rnn.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Chưa có model: {path.relative_to(ROOT)}")
            with np.load(path, allow_pickle=False) as p:
                W_xh, W_hh = p["W_xh"], p["W_hh"]
                b_h, W_hy, b_y = p["b_h"], p["W_hy"], p["b_y"]
            h = np.zeros((1, W_hh.shape[0]), dtype=np.float32)
            for t in range(x.shape[1]):
                h = np.tanh(x[:, t, :] @ W_xh + h @ W_hh + b_h)
            return float((h @ W_hy + b_y).reshape(-1)[0])
        if framework == "Keras":
            path = base / "keras_rnn.keras"
            if not path.is_file():
                raise FileNotFoundError(f"Chưa có model: {path.relative_to(ROOT)}")
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError("Thiếu TensorFlow. Cài bằng pip install tensorflow.") from exc
            model = tf.keras.models.load_model(path, compile=False)
            self.models[cache_key] = lambda a: float(np.asarray(model.predict(a, verbose=0)).reshape(-1)[0])
            return self.models[cache_key](x)
        if framework == "PyTorch":
            path = base / "pytorch_rnn.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Chưa có model: {path.relative_to(ROOT)}")
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Thiếu PyTorch. Cài bằng pip install torch.") from exc

            class TorchRNNRegressor(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.rnn = torch.nn.RNN(n_features, hidden_size, num_layers=1, nonlinearity="tanh", bias=True, batch_first=True)
                    self.dropout = torch.nn.Dropout(0.0)
                    self.fc = torch.nn.Linear(hidden_size, 1)

                def forward(self, values):
                    output, hidden = self.rnn(values)
                    return self.fc(self.dropout(output[:, -1, :])), hidden, output

            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model = TorchRNNRegressor()
            model.load_state_dict(state)
            model.eval()
            torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))

            def infer(a):
                with torch.inference_mode():
                    out, _, _ = model(torch.tensor(a, dtype=torch.float32))
                return float(out.cpu().numpy().reshape(-1)[0])

            self.models[cache_key] = infer
            return infer(x)
        raise ValueError("Framework không hợp lệ. Chọn Scratch, Keras hoặc PyTorch.")

    def predict(self, request):
        if np is None or joblib is None:
            raise RuntimeError("Cài các thư viện cần thiết bằng: pip install numpy joblib scikit-learn")
        dataset_key = str(request.get("dataset", ""))
        framework = str(request.get("framework", ""))
        if dataset_key not in self.datasets:
            raise ValueError("Dataset chưa có trong notebook artifacts.")
        if framework not in FRAMEWORKS:
            raise ValueError("Mô hình phải là Scratch, Keras hoặc PyTorch.")
        d = self.datasets[dataset_key]
        if d.get("x_scaler") is None or d.get("y_scaler") is None:
            raise RuntimeError("Không nạp được scaler. Hãy chạy cell export cuối notebook.")
        try:
            raw = np.asarray(request.get("window"), dtype=np.float32)
        except Exception as exc:
            raise ValueError("Input phải là ma trận số.") from exc
        expected = (d["lookback"], len(d["features"]))
        if raw.shape != expected:
            raise ValueError(f"Input sai kích thước: cần {expected[0]} hàng × {expected[1]} cột; nhận được {raw.shape}.")
        if not np.isfinite(raw).all():
            raise ValueError("Input chứa ô trống, NaN hoặc giá trị vô hạn.")
        x_scaled = d["x_scaler"].transform(raw).astype(np.float32)[None, :, :]
        y_scaled = self._predict_model(dataset_key, framework, x_scaled)
        prediction = float(d["y_scaler"].inverse_transform(np.array([[y_scaled]], dtype=np.float32))[0, 0])
        summary = next((row for row in d.get("summary", []) if row.get("framework") == framework), {})
        return {
            "dataset": dataset_key,
            "dataset_name": d.get("display_name", dataset_key),
            "framework": framework,
            "prediction": prediction,
            "target_unit": d.get("target_unit", ""),
            "forecast_horizon": d.get("forecast_horizon", "next step"),
            "latest_observed_target": d.get("latest_observed_target"),
            "metrics": {k: summary.get(k) for k in ("MAE", "RMSE", "MAPE_percent", "R2")},
            "lookback": d["lookback"],
            "features": d["features"],
        }


ARTIFACTS = NotebookArtifacts()


class Handler(BaseHTTPRequestHandler):
    server_version = "RNNForecastDemo/1.0"

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value):
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if HTML_FILE.is_file():
                body = HTML_FILE.read_bytes()
            else:
                body = PAGE.encode("utf-8")
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path in ("/static/style.css", "/static/app.js"):
            asset_name = Path(self.path).name
            asset_path = STATIC_DIR / asset_name
            if not asset_path.is_file():
                self._json(404, {"error": f"Thiếu file giao diện: {asset_path}"})
                return
            content_type = "text/css; charset=utf-8" if asset_name.endswith(".css") else "application/javascript; charset=utf-8"
            self._send(200, asset_path.read_bytes(), content_type)
        elif self.path == "/api/bootstrap":
            self._json(200, ARTIFACTS.bootstrap())
        elif self.path == "/health":
            self._json(200, {"status": "ok", "artifacts_ready": ARTIFACTS.bootstrap()["ready"]})
        else:
            self._json(404, {"error": "Không tìm thấy đường dẫn."})

    def do_POST(self):
        if self.path != "/api/predict":
            self._json(404, {"error": "Không tìm thấy API."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise ValueError("Kích thước yêu cầu không hợp lệ.")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            self._json(200, ARTIFACTS.predict(request))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"RNN Forecast Lab: {url}")
    print(f"Project root: {ROOT}")
    if ARTIFACTS.bootstrap()["missing"]:
        print("Notebook artifacts chưa đầy đủ. Giao diện vẫn mở; xem banner trên trang để biết file còn thiếu.")
    else:
        print("Artifacts đã sẵn sàng cho inference.")
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng web server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
