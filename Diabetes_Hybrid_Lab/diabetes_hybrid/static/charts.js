const payload = document.querySelector("#chart-data");
if (payload) {
  const {history, diagnostics} = JSON.parse(payload.textContent);
  const area = document.querySelector("#model-charts");
  const colors = ["#087569", "#ac521c"];

  function lineChart(title, series, xLabel, yLabel) {
    const box = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = title;
    box.append(caption);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 400 207");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", title);
    function shape(kind, attributes, label) {
      const element = document.createElementNS(svg.namespaceURI, kind);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      if (label) element.textContent = label;
      svg.append(element);
    }
    const xs = series.flatMap(item => item.x);
    const ys = series.flatMap(item => item.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xAt = value => 48 + 334 * (value - xMin) / (xMax - xMin || 1);
    const yAt = value => 157 - 131 * (value - yMin) / (yMax - yMin || 1);
    shape("path", {d:"M48 24V157H382", stroke:"#7d9994", fill:"none"});
    shape("text", {x:48,y:17,fill:"#4e6666","font-size":11}, yLabel);
    shape("text", {x:48,y:191,fill:"#4e6666","font-size":11}, xLabel);
    shape("text", {x:8,y:31,fill:"#4e6666","font-size":10}, yMax.toFixed(2));
    shape("text", {x:8,y:159,fill:"#4e6666","font-size":10}, yMin.toFixed(2));
    shape("text", {x:48,y:173,fill:"#4e6666","font-size":10}, String(xMin.toFixed(2)));
    shape("text", {x:348,y:173,fill:"#4e6666","font-size":10}, String(xMax.toFixed(2)));
    series.forEach((item,index) => {
      shape("polyline", {points:item.x.map((x,i)=>`${xAt(x)},${yAt(item.y[i])}`).join(" "),
                         fill:"none",stroke:colors[index],"stroke-width":2});
      shape("text", {x:178+index*94,y:191,fill:colors[index],"font-size":10}, item.label);
    });
    box.append(svg);
    area.append(box);
  }

  const epoch = history.map(row => row.epoch);
  const trainVal = (train,val) => [{x:epoch,y:history.map(row=>row[train]),label:translateDemo("Huấn luyện")},
                                    {x:epoch,y:history.map(row=>row[val]),label:translateDemo("Xác thực")}];
  lineChart(translateDemo("Hàm mất mát"),trainVal("train_weighted_loss","val_weighted_loss"),"Epoch",translateDemo("Loss có trọng số"));
  lineChart(translateDemo("Độ chính xác"),trainVal("train_accuracy","val_accuracy"),"Epoch",translateDemo("Tỷ lệ đúng"));
  for (const [key,title,xLabel,yLabel] of [
    ["roc",translateDemo("Đường ROC"),translateDemo("Tỷ lệ dương tính giả"),translateDemo("Tỷ lệ dương tính thật")],
    ["pr",translateDemo("Đường precision–recall"),"Recall","Precision"],
    ["calibration",translateDemo("Đồ thị hiệu chuẩn"),translateDemo("Điểm trung bình"),translateDemo("Tần suất quan sát")],
  ]) {
    lineChart(title,["MLP","Logistic Regression"].map((name,i)=>({...diagnostics.curves[name][key],
              label:i===0?"MLP":"Logistic"})),xLabel,yLabel);
  }
  const histogram = diagnostics.curves.MLP.score_histogram;
  lineChart(translateDemo("Phân bố điểm của MLP"),[{x:histogram.edges.slice(0,-1),y:histogram.counts,label:"MLP"}],
            translateDemo("Điểm lớp 2"),translateDemo("Số hồ sơ trong khoảng"));

  const importance = document.createElement("figure");
  const title = document.createElement("figcaption");
  title.textContent = translateDemo("Mức giảm Macro-F1 khi xáo trộn biến");
  importance.append(title);
  const list = document.createElement("ol");
  for (const row of [...diagnostics.permutation].sort((a,b)=>b.mean_drop-a.mean_drop).slice(0,7)) {
    const item = document.createElement("li");
    item.textContent = `${row.feature} · ${row.mean_drop.toFixed(4)}`;
    list.append(item);
  }
  importance.append(list);
  area.append(importance);
}

