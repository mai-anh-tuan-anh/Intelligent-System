# Kịch bản demo Diabetes Hybrid Lab

Buổi demo đi theo ba phần trên bảng. Cho thầy xem cách nhiều nguồn trở thành một kho Neo4j, truy vết một câu hỏi từ RAG đến kho đó, rồi chạy luồng đa tác tử kết hợp chatbot với mô hình học sâu. Có thể dành khoảng 2–3 phút mỗi phần.

## Chuẩn bị

Trên máy đang dùng Neo4j Desktop, kiểm tra instance **Diabetes** hiện **Running**. Mở terminal VS Code tại thư mục chứa `app.py` và chạy `python app.py`. Cấu hình kết nối được đọc từ máy. Mở [web demo Desktop](http://127.0.0.1:8012/#demo), [trang kiểm tra kết nối](http://127.0.0.1:8012/api/graph) và dùng mục **Query** trong Neo4j Desktop để xem đồ thị.

Nếu dùng cấu hình Neo4j portable, mở PowerShell và chạy từ thư mục project.

```powershell
Set-Location -LiteralPath 'C:\Users\ADMIN\Downloads\Intelligent System\Diabetes_Hybrid_Lab\diabetes_hybrid'
./start_demo.ps1
```

Giữ cửa sổ này mở. Script khởi động Neo4j cục bộ nếu cần, sau đó chạy `app.py`. Mở [web demo](http://127.0.0.1:8011/) và [Neo4j Browser](http://127.0.0.1:7476/browser/). Trong Neo4j Browser, kết nối tới `bolt://127.0.0.1:7689`, tài khoản `neo4j`, database `neo4j`. Mật khẩu trên máy này được lưu trong tệp cục bộ `%LOCALAPPDATA%\DiabetesHybridRuntime\neo4j-password.txt`. Chuẩn bị phần đăng nhập trước giờ báo cáo.

Kiểm tra [trạng thái Neo4j](http://127.0.0.1:8011/api/graph). Mục `ready` cần có giá trị `true`. Nếu vừa chuyển sang máy khác, dùng hướng dẫn thiết lập ở [README](../README.md). Khi cần ảnh dự phòng, mở [ảnh Neo4j Browser](../demo/04_neo4j_browser.png) và [ảnh demo kết hợp](../demo/02_combined_vinmec.png).

## 1. Cho thầy xem KB Neo4j và cách tổng hợp

**Mở các tệp theo thứ tự này.**

1. [vinmec.json](../knowledge_base/sources/vinmec.json), [who.json](../knowledge_base/sources/who.json) và [codebook.json](../knowledge_base/sources/codebook.json). Mỗi tệp lưu một URL cùng metadata, đoạn tri thức đã tuyển chọn bằng hai ngôn ngữ, và quan hệ có bằng chứng nếu có. Chỉ vào `p02` trong tệp Vinmec và `p12` trong tệp CDC. PDF CDC mô tả dữ liệu khảo sát BRFSS và được lưu dưới dạng URL tới trang 21 cùng đoạn diễn giải.
2. [build_kb.py](../knowledge_base/build_kb.py). Chỉ vào `collect()`, `cypher_statements()` và `build()`. `collect()` đọc từng JSON, đối chiếu khái niệm với [concepts.json](../knowledge_base/concepts.json), kiểm tra trùng ID/URL và kiểm tra bằng chứng của cạnh. `cypher_statements()` sinh ràng buộc ID và các lệnh `UNWIND`/`MERGE`. `build()` ghi bản tổng hợp [kb_diabetes.json](../knowledge_base/kb_diabetes.json) và tệp [kb_diabet.cypher](../knowledge_base/cypher/kb_diabet.cypher).
3. [kb_diabet.cypher](../knowledge_base/cypher/kb_diabet.cypher). Tìm `vinmec`, `p02` và `FROM_SOURCE` để chỉ đúng đoạn chuyển từ tệp nguồn sang nút và cạnh Neo4j.

Nguồn ví dụ gồm [bài Vinmec](https://www.vinmec.com/vie/bai-viet/tieu-chuan-chan-doan-tieu-duong-type-1-vi), [chuyên trang tiểu đường của WHO](https://www.who.int/news-room/fact-sheets/detail/diabetes) và [PDF CDC BRFSS, trang 21](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf#page=21).

Trong PowerShell thứ hai, chạy từ thư mục project để minh họa quá trình tạo và nạp kho.

```powershell
python -m knowledge_base.build_kb
python -m knowledge_base.build_kb --check
python -m knowledge_base.seed
```

Kết quả hiện tại là **22 nguồn, 27 đoạn tri thức, 21 khái niệm và 112 quan hệ**. Lệnh seed dùng cùng bộ lệnh mà builder xuất ra tệp Cypher. `MERGE` dựa trên ID và quan hệ để có thể chạy lại với cùng bộ nguồn.

Nếu thầy yêu cầu chạy trực tiếp **chính tệp Cypher**, dùng lệnh sau trong môi trường đã có Cypher Shell. Nhập mật khẩu khi được hỏi. [Tài liệu Cypher Shell](https://neo4j.com/docs/operations-manual/current/cypher-shell/)

```powershell
cypher-shell -a bolt://127.0.0.1:7689 -u neo4j -d neo4j -f knowledge_base/cypher/kb_diabet.cypher
```

**Nếu thầy hỏi lưu ở đâu**, mở `knowledge_base/cypher/kb_diabet.cypher` trong project. Đây là tệp để tái tạo kho. Dữ liệu Neo4j đang chạy nằm ở database `neo4j` của bản Neo4j cục bộ trên máy này, dưới `%LOCALAPPDATA%\DiabetesHybridRuntime\neo4j-community-5.26.33\data\databases\neo4j`. Trong demo, xem dữ liệu qua Neo4j Browser.

**Chuyển sang Neo4j Browser.** Chạy từng truy vấn trong [examples.cypher](../knowledge_base/cypher/examples.cypher). Bắt đầu bằng truy vấn số 5 để thấy ba nút `Concept`, `Passage` và `Source` nối với nhau. Bấm từng nút để xem thuộc tính, hoặc mở chế độ bảng để thấy URL. Nếu thầy hỏi quan hệ ngữ nghĩa, chạy truy vấn số 2 để thấy cạnh `CLASSIFIED_WITH` và đoạn trích làm bằng chứng. Truy vấn số 4 mở đường dẫn tới PDF CDC, còn truy vấn số 6 thống kê các nguồn đã gộp theo nhà xuất bản.

```cypher
MATCH path = (c:Concept {id:'type1'})<-[:ABOUT]-(p:Passage {id:'p02'})
             -[:FROM_SOURCE]->(s:Source {id:'vinmec'})
RETURN path, p.id AS passage_id, p.text_vi AS passage, s.url AS source_url;
```

## 2. Cho thầy xem RAG truy xuất từ KB và tạo câu trả lời

1. Mở [web demo](http://127.0.0.1:8011/#demo). Chọn **Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?** rồi bấm **Chỉ hỏi đáp**.
2. Mở **Nguồn và bằng chứng**. Đối chiếu đoạn Vinmec với `p02` vừa mở trong Neo4j. Trong lần kiểm tra, kết quả xếp theo thứ tự `p03` (NIDDK), `p02` (Vinmec), `p22` (CDC).
3. Mở [core/rag.py](../core/rag.py), tìm `GraphRAG.answer()` và lệnh `self.query(RETRIEVE, anchors=anchors, language=language)`. Python chuẩn hóa câu hỏi, tìm ID khái niệm rồi truyền các ID này vào biến `$anchors` trong [retrieve.cypher](../knowledge_base/cypher/retrieve.cypher). Neo4j trả về đoạn trích, nguồn và quan hệ. Python xếp hạng theo TF-IDF ký tự, khớp chủ đề và nhóm nguồn. `ExplanationAgent.compose()` kiểm tra nguồn và chọn đoạn đứng đầu làm câu trả lời.
4. Nếu thầy muốn xem dữ liệu trung gian, chạy đoạn sau trong PowerShell. Đối chiếu `anchors` và `passage_id` với nút vừa xem trong Neo4j.

```powershell
$question = 'Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?'
$body = @{ mode = 'qa'; question = $question } | ConvertTo-Json
$run = Invoke-RestMethod -Uri 'http://127.0.0.1:8011/api/run' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
$run.knowledge.anchors
$run.knowledge.evidence | Select-Object @{n='passage_id';e={$_.id}}, title, score, @{n='url';e={$_.source.url}}
```

**Cách mô tả khi nói.** “Kho được nạp vào Neo4j bằng Cypher. Khi có câu hỏi, RAG gửi truy vấn đọc đến kho, nhận đoạn trích và URL, rồi trả lời theo đoạn phù hợp nhất. Bản hiện tại dùng câu trả lời trích xuất có dẫn nguồn. Nó chưa tích hợp mô hình ngôn ngữ sinh văn bản.”

## 3. Demo hệ đa tác tử kết hợp Deep Learning và tri thức

Mở [sơ đồ luồng](../artifacts/figures/01_agent_flow.png) hoặc [sơ đồ tuần tự](../artifacts/figures/09_sequence_diagram.png) trước khi bấm chạy. Chọn **Hồ sơ B**, giữ **MLP gốc từ Assignment 4**, điền lại câu hỏi về xét nghiệm típ 1 và chọn **Chạy cả hai nhánh**.

- **Nhánh mô hình.** Coordinator gọi Data Agent kiểm tra 21 trường, rồi Prediction Agent chạy checkpoint BRFSS. Rule Agent kiểm tra khoảng cách giữa hai điểm cao nhất.
- **Nhánh tri thức.** Rule Agent kiểm tra phạm vi câu hỏi, Knowledge Agent truy xuất Neo4j và Explanation Agent tổng hợp kết quả có nguồn.
- **Báo cáo.** Chỉ vào ba điểm mô hình, câu trả lời có nguồn và bảng trace. Monitoring ghi trạng thái, thời gian, phiên bản và mã bằng chứng với cùng mã yêu cầu. Điểm mô hình mô tả nhãn khảo sát BRFSS trên hồ sơ giả lập.

Nếu còn thời gian, nhập **Tiểu đường thai kỳ là gì?** rồi **Còn xét nghiệm thì sao?** để minh họa cách giữ chủ đề trong hội thoại. Có thể mở thêm [notebook tiếng Anh](../notebooks/Diabetes_Hybrid_Lab.ipynb), [bộ hỏi đáp tiếng Anh](PRESENTATION_QA.md) và [kết quả kiểm thử](../artifacts/demo_verification.json).

## Mở nhanh theo nội dung

| Thầy muốn xem | Vị trí trong project |
| --- | --- |
| Nguồn gốc và đoạn trích | [knowledge_base/sources/](../knowledge_base/sources/) |
| Mã hợp nhất và file nạp Neo4j | [build_kb.py](../knowledge_base/build_kb.py), [kb_diabet.cypher](../knowledge_base/cypher/kb_diabet.cypher) |
| Truy vấn chatbot chạy thật | [retrieve.cypher](../knowledge_base/cypher/retrieve.cypher), [rag.py](../core/rag.py) |
| Điều phối và báo cáo | [agents.py](../core/agents.py), [sơ đồ tuần tự](../artifacts/figures/09_sequence_diagram.png) |
| Dữ liệu, mô hình, thực nghiệm | [data/](../data/), [artifacts/](../artifacts/), [experiments/](../experiments/) |
| Báo cáo đã lưu trong project | [project_01_04_anhmat.pdf](project_01_04_anhmat.pdf) |
