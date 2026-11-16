# Đáp án vấn đáp Phát triển hệ thống thông minh

Tài liệu đề có tiêu đề **50 Questions**, nhưng nội dung đánh số từ **câu 1 đến câu 51**. Bộ đáp án này giữ nguyên thứ tự và trả lời cả câu 51. Các câu thực hành dùng kết quả đã lưu của Diabetes Hybrid Lab và các bài Assignment trong workspace.

## Phần 1. Machine Learning và Deep Learning

### Câu 1. Machine Learning là gì? Phân biệt ba cách học và cho ví dụ

Machine Learning là cách xây dựng chương trình có thể rút ra quy luật từ dữ liệu hoặc trải nghiệm để thực hiện một nhiệm vụ trên đầu vào mới.

- **Học có giám sát** dùng cặp đầu vào và nhãn, ví dụ học từ hồ sơ BRFSS có nhãn khảo sát để phân loại một hồ sơ khác.
- **Học không giám sát** tìm cấu trúc trong dữ liệu chưa gắn nhãn, ví dụ gom nhóm khách hàng theo tần suất và giá trị giao dịch.
- **Học tăng cường** học cách chọn hành động từ phần thưởng qua nhiều bước, ví dụ robot học đường đi trong môi trường mô phỏng để tới đích với ít va chạm.

Phân biệt bằng tín hiệu học. Ba trường hợp lần lượt dùng **nhãn**, **cấu trúc của dữ liệu** và **phần thưởng qua tương tác**.

### Câu 2. Quy trình hoàn chỉnh để phát triển mô hình dự đoán

1. Xác định mục tiêu, đơn vị quan sát, nhãn và chỉ số đánh giá.
2. Thu thập dữ liệu, kiểm tra nguồn, dữ liệu thiếu, dòng trùng và phân bố nhãn.
3. Chia train, validation và test. Chia theo thời gian với dữ liệu chuỗi, hoặc theo người/hồ sơ khi nhiều dòng có liên quan.
4. Xử lý thiếu dữ liệu, mã hóa, chuẩn hóa và tạo đặc trưng. Fit những bước học từ dữ liệu trên **tập train**.
5. Huấn luyện baseline và các mô hình ứng viên trên train. Dùng validation để chọn tham số, ngưỡng hoặc checkpoint.
6. Đánh giá cuối trên test và lưu cả mô hình, bộ tiền xử lý, schema đầu vào và phiên bản.

Trong BRFSS, em nhóm các dòng có đặc trưng trùng nhau trước khi chia tập để giảm rò rỉ giữa train và test.

### Câu 3. Phân loại và hồi quy khác nhau thế nào?

**Phân loại** dự đoán một nhãn rời rạc, ví dụ ba lớp khảo sát BRFSS. **Hồi quy** dự đoán một giá trị số liên tục, ví dụ giá đóng cửa AAPL phiên kế tiếp hoặc giá bán lại một căn hộ. Nếu mục tiêu chứng khoán đổi thành “tăng hay giảm”, bài toán trở thành phân loại. Với phân loại em có thể dùng macro-F1 và recall. Với hồi quy em dùng MAE, RMSE hoặc R² tùy mục tiêu.

### Câu 4. Mạng nơ-ron nhân tạo gồm những thành phần nào?

Tầng đầu vào nhận vector đặc trưng. Mỗi tầng ẩn tính `z = Wx + b` rồi áp dụng hàm kích hoạt `a = f(z)` để học quan hệ phi tuyến. Trọng số `W` cho biết mức đóng góp của đầu vào vào từng nơ-ron. Bias `b` dịch chuyển giá trị trước kích hoạt. Tầng đầu ra được thiết kế theo bài toán, ví dụ một giá trị cho hồi quy hoặc ba logit cho phân loại BRFSS. Trong project, MLP có kích thước **21 → 64 → 32 → 3**, với ReLU ở các tầng ẩn.

### Câu 5. Forward propagation và backpropagation

**Forward propagation** đưa dữ liệu qua từng tầng để tạo dự đoán, sau đó tính loss bằng nhãn thật. **Backpropagation** dùng quy tắc dây chuyền để tính đạo hàm của loss đối với từng tham số, đi từ tầng cuối về đầu. Ví dụ, nếu `z = wx + b` và `a = f(z)` thì `∂L/∂w = (∂L/∂a) × f′(z) × x`. Optimizer dùng gradient để cập nhật tham số. Dạng gradient descent đơn giản là `w mới = w cũ − η × ∂L/∂w`, với `η` là learning rate.

### Câu 6. Loss function là gì? So sánh MSE, BCE và CCE

Loss đo mức sai giữa đầu ra mô hình và mục tiêu. Khi huấn luyện, optimizer tìm tham số làm giảm loss.

| Loss | Dạng cơ bản | Ví dụ sử dụng |
|---|---|---|
| **MSE** | Trung bình của `(y − ŷ)²` | Hồi quy giá nhà. Sai số lớn bị phạt mạnh do bình phương. |
| **Binary Cross-Entropy** | `−[y log(p) + (1−y) log(1−p)]` | Phân loại nhị phân, ví dụ thư rác. |
| **Categorical Cross-Entropy** | `−Σ yᵢ log(pᵢ)` với các lớp loại trừ nhau | BRFSS ba lớp. Với nhãn thật dạng chỉ số, loss là `−log(p_lớp_thật)`. |

Trong PyTorch, `CrossEntropyLoss` nhận logit thô và nhãn lớp. Khi suy luận, project mới dùng softmax để hiển thị ba điểm.

### Câu 7. Gradient Descent và các biến thể

Gradient Descent cập nhật theo hướng đối diện gradient, với độ lớn bước chịu ảnh hưởng của learning rate.

- **Batch** tính gradient trên toàn tập train cho mỗi lần cập nhật. Gradient ổn định hơn nhưng mỗi bước tốn chi phí cao.
- **Stochastic** dùng một mẫu mỗi bước. Cập nhật nhanh và nhiều nhiễu.
- **Mini-batch** dùng một nhóm nhỏ. Cách này cân bằng độ nhiễu và hiệu quả tính toán trên phần cứng.

Learning rate quá nhỏ làm quá trình học chậm. Quá lớn có thể làm loss dao động hoặc phân kỳ. AdamW trong project dùng gradient mini-batch và điều chỉnh bước cập nhật cho từng tham số.

### Câu 8. Overfitting, underfitting và cách xử lý

**Underfitting** xảy ra khi mô hình chưa học đủ quy luật, nên kết quả thường kém trên cả train lẫn validation. **Overfitting** thường thể hiện qua kết quả train tốt nhưng validation kém do mô hình học cả nhiễu hoặc đặc thù của tập train. Train dùng để học trọng số, validation dùng để chọn cấu hình và checkpoint, test dùng để ước lượng khả năng tổng quát hóa sau khi đã chọn.

Regularization như L2 hạn chế trọng số quá lớn. Dropout bỏ ngẫu nhiên một phần đầu ra của nơ-ron khi huấn luyện để giảm phụ thuộc vào những đường truyền cụ thể. Early stopping giữ checkpoint tốt nhất theo validation và có thể dừng khi chỉ số không cải thiện sau số epoch đã định.

### Câu 9. Đánh giá mô hình phân loại như thế nào?

Confusion matrix đếm nhãn thật theo nhãn dự đoán. Với một lớp được coi là dương, ta có TP, FP, FN, TN.

- **Accuracy** là `(TP + TN)/N`, tỷ lệ dự đoán đúng.
- **Precision** là `TP/(TP + FP)`, tỷ lệ dự đoán dương thực sự dương.
- **Recall** là `TP/(TP + FN)`, tỷ lệ mẫu dương được phát hiện.
- **F1** là `2 × precision × recall / (precision + recall)`. **Macro-F1** lấy trung bình F1 của các lớp với trọng số bằng nhau.

Trong tập test BRFSS chia theo nhóm, baseline luôn chọn lớp phổ biến nhất đạt **84,11% accuracy** nhưng recall lớp tiền tiểu đường bằng **0**. Vì vậy cần xem confusion matrix, recall từng lớp và macro-F1. [Kết quả BRFSS](../artifacts/experiment.json)

### Câu 10. So sánh mô hình phân loại, hồi quy và những khó khăn khi triển khai

Ứng dụng phân loại trả nhãn và có thể trả thêm điểm theo lớp. Ứng dụng hồi quy trả một giá trị cùng đơn vị với mục tiêu, ví dụ giá nhà tính bằng SGD. Khâu triển khai cần kiểm tra schema và đơn vị, chạy đúng preprocessing của lúc huấn luyện, nạp checkpoint phù hợp, xử lý dữ liệu sai, theo dõi độ trễ và quản lý phiên bản. Đầu vào thực tế có thể thay đổi theo thời gian, do đó cần theo dõi hiệu năng sau triển khai.

Ví dụ trong project, Flask kiểm tra dữ liệu hồ sơ, nạp checkpoint đã lưu và trả lỗi có nội dung khi đầu vào sai. Nhánh Neo4j có thể trả trạng thái chưa kết nối trong khi nhánh MLP vẫn trả điểm.

## Phần 2. CNN, RNN và LSTM

### Câu 11. Kiến trúc CNN gồm những gì?

CNN dùng các lớp tích chập để quét bộ lọc trên đầu vào. Mỗi bộ lọc học một mẫu và tạo một **feature map** thể hiện mức phản ứng của mẫu ở từng vị trí. Hàm kích hoạt như ReLU tạo phi tuyến. Pooling có thể giảm kích thước không gian, ví dụ max pooling lấy giá trị lớn nhất trong một vùng. Sau các khối trích đặc trưng, lớp phân loại dùng các feature map đã tổng hợp để tạo điểm theo lớp. Một CNN có thể dùng tích chập stride lớn hoặc global average pooling tùy thiết kế.

### Câu 12. Tích chập hoạt động như thế nào?

Tại mỗi vị trí, kernel lấy tổng có trọng số của vùng đầu vào mà nó phủ và cộng bias. Với ảnh nhiều kênh, mỗi bộ lọc trải qua các kênh đầu vào. Kernel size quy định vùng quan sát cục bộ. Stride quy định bước trượt. Padding bổ sung giá trị ở mép để điều chỉnh kích thước và mức phủ biên. Số bộ lọc quyết định số kênh đầu ra.

Với dilation bằng 1, một chiều có kích thước đầu ra `⌊(N + 2P − K)/S⌋ + 1`. Ví dụ, ảnh 32 × 32 dùng kernel 3 × 3, padding 1, stride 1 và 16 bộ lọc cho đầu ra 32 × 32 × 16. [Tài liệu Conv2d của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

### Câu 13. Vì sao CNN hiệu quả với ảnh?

Pixel lân cận thường liên quan với nhau. Kết nối cục bộ tận dụng đặc điểm đó, còn chia sẻ trọng số giúp cùng một bộ lọc phát hiện mẫu ở nhiều vị trí và giảm số tham số. Các tầng đầu thường học cạnh và họa tiết đơn giản. Các tầng sâu có vùng quan sát lớn hơn, kết hợp nhiều đặc trưng để biểu diễn bộ phận hoặc hình dạng. Đây là xu hướng thường gặp, còn đặc trưng cụ thể phụ thuộc dữ liệu và mục tiêu huấn luyện.

### Câu 14. Quy trình xây dựng ứng dụng phân loại ảnh bằng CNN

Xác định lớp, thu thập và gán nhãn ảnh, loại rò rỉ như ảnh trùng giữa các tập. Chuyển ảnh về kích thước, số kênh và cách chuẩn hóa thống nhất. Áp dụng augmentation trên train nếu biến đổi vẫn giữ nhãn. Huấn luyện CNN trên train, dùng validation chọn cấu hình và test để báo cáo. Khi triển khai, API kiểm tra tệp ảnh và tiền xử lý giống lúc huấn luyện, đưa model về chế độ inference rồi trả nhãn và điểm. Với ảnh đầu vào khó nhận diện, có thể thêm cơ chế báo độ tin cậy thấp theo ngưỡng được đánh giá.

### Câu 15. RNN và hidden state là gì?

RNN xử lý chuỗi bằng cách cập nhật trạng thái ẩn ở từng bước. Công thức đơn giản là `hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)`. `xₜ` là đầu vào hiện tại, `hₜ₋₁` tóm tắt thông tin trước đó. Các trọng số được dùng lại qua các bước thời gian. Dự đoán có thể lấy từ trạng thái cuối, ví dụ dự báo giá kế tiếp từ một cửa sổ giá, hoặc lấy ở mỗi bước, ví dụ gán nhãn từng phần tử chuỗi.

### Câu 16. Vanishing gradient và exploding gradient trong RNN

Backpropagation through time nhân nhiều ma trận đạo hàm qua các bước. Nếu các phép nhân liên tục làm độ lớn gradient giảm, gradient về các bước xa có thể gần 0. Nếu độ lớn tăng liên tục, gradient có thể bùng nổ và gây cập nhật mất ổn định. Trường hợp đầu làm RNN khó học quan hệ dài hạn. Gradient clipping thường dùng để khống chế gradient bùng nổ. LSTM và GRU tạo đường truyền thông tin có kiểm soát để hỗ trợ học quan hệ dài hơn.

### Câu 17. LSTM hoạt động như thế nào?

LSTM có hai trạng thái. **Cell state** `cₜ` lưu bộ nhớ và **hidden state** `hₜ` biểu diễn đầu ra ở bước hiện tại.

- **Forget gate** `fₜ` điều chỉnh lượng bộ nhớ cũ được giữ.
- **Input gate** `iₜ` điều chỉnh lượng thông tin ứng viên `gₜ` được ghi.
- **Output gate** `oₜ` điều chỉnh phần bộ nhớ được đưa ra trạng thái ẩn.

Hai công thức chính là `cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ` và `hₜ = oₜ ⊙ tanh(cₜ)`, trong đó `⊙` là nhân từng phần tử. Các cổng sigmoid có giá trị từ 0 đến 1. Đường cộng trong cell state giúp thông tin và gradient có thể truyền xa hơn khi forget gate giữ bộ nhớ. [Tài liệu LSTM của PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

### Câu 18. So sánh RNN, LSTM và CNN. Ứng dụng vào dự báo cổ phiếu

CNN mạnh ở mẫu cục bộ và xử lý song song hiệu quả, phù hợp ảnh hoặc chuỗi có mẫu ngắn. RNN biểu diễn thứ tự bằng hidden state, tương đối gọn nhưng khó giữ phụ thuộc dài. LSTM thêm cổng và bộ nhớ để kiểm soát thông tin dài hạn, đổi lại có nhiều tham số và tính toán hơn.

Với cổ phiếu, RNN hoặc LSTM có thể nhận cửa sổ giá theo thời gian và dự báo giá kế tiếp. Cần chia theo thời gian, tránh dùng dữ liệu tương lai trong preprocessing, và so sánh với baseline lấy giá cuối. Trong Assignment 6, baseline này thực tế có RMSE thấp hơn các RNN đã thử, nên việc chọn RNN cần dựa trên số liệu. [Kết quả Assignment 6](../../../assignment%206/DATA/outputs/metrics/all_experiments_summary.csv)

## Phần 3. Knowledge Graph Neo4j và RAG

### Câu 19. Knowledge Graph là gì?

Knowledge Graph biểu diễn tri thức dưới dạng thực thể và quan hệ có ý nghĩa. Trong đồ thị của em, `type1` là một khái niệm, `p02` là một đoạn bằng chứng và `vinmec` là nguồn. Cạnh `ABOUT` cho biết đoạn nói về chủ đề nào. Cạnh `FROM_SOURCE` cho biết đoạn đến từ đâu. Thuộc tính mô tả nút hoặc cạnh, ví dụ ID, tiêu đề và URL của nguồn. Nhờ đồ thị, em có thể truy từ một chủ đề về đoạn trích và xuất xứ của nó.

### Câu 20. Biểu diễn tri thức bằng bộ ba và cho ít nhất năm ví dụ

Bộ ba gồm **chủ thể**, **quan hệ** và **đối tượng**. Các bộ ba sau tương ứng với quan hệ trong project:

1. `(p02, ABOUT, type1)`
2. `(p02, FROM_SOURCE, vinmec)`
3. `(p03, ABOUT, type1)`
4. `(p03, FROM_SOURCE, type1)` với đối tượng ở đây là nút `Source` NIDDK mang ID `type1`
5. `(p12, FROM_SOURCE, codebook)`

Khi demo, nên gọi kèm nhãn nút, chẳng hạn `Concept:type1` và `Source:type1`, vì ID ở hai loại nút có thể giống nhau. [Tệp Cypher đã sinh](../knowledge_base/cypher/kb_diabet.cypher)

### Câu 21. Neo4j và mô hình property graph

Neo4j là hệ quản trị cơ sở dữ liệu đồ thị. Nó biểu diễn một nút bằng nhãn như `Concept` và các thuộc tính khóa–giá trị như `id`. Quan hệ có điểm đầu, điểm cuối, hướng, loại và có thể có thuộc tính riêng như `passage_id`. Cypher cho phép mô tả mẫu nút và cạnh cần tìm. Chỉ mục hoặc ràng buộc duy nhất có thể hỗ trợ tìm nút xuất phát theo ID, rồi truy vấn đi qua các quan hệ phù hợp.

### Câu 22. Dùng Cypher tạo một Patient, một Disease và truy xuất liên kết

Cypher dùng `MATCH` để tìm mẫu, `WHERE` để lọc, `RETURN` để trả trường. `CREATE` tạo mới. `MERGE` tìm hoặc tạo mẫu theo điều kiện. Ví dụ minh họa sau dùng hồ sơ giả lập và chạy trong database thực hành riêng. [Tài liệu MERGE của Neo4j](https://neo4j.com/docs/cypher-manual/current/clauses/merge/)

```cypher
MERGE (p:Patient {id: 'synthetic-01'})
MERGE (d:Disease {id: 'type2'})
SET d.name = 'Type 2 diabetes'
MERGE (p)-[:RECORDED_DIAGNOSIS {simulated: true}]->(d)
RETURN p.id AS patient_id, d.name AS disease;
```

```cypher
MATCH (p:Patient {id: 'synthetic-01'})-[r:RECORDED_DIAGNOSIS]->(d:Disease)
RETURN p.id AS patient_id, type(r) AS relation, d.name AS disease, r.simulated;
```

Trong project thật, kho của chatbot lưu `Concept`, `Passage` và `Source`. Hồ sơ đưa vào MLP được xử lý để tạo báo cáo của lượt chạy.

### Câu 23. Thiết kế đồ thị cho miền tiểu đường

Một mô hình mở rộng có thể dùng `Patient`, `Symptom`, `RiskFactor`, `Measurement`, `Disease`, `Recommendation` và `Source`. Liên kết ví dụ gồm `Patient → HAS_REPORTED_SYMPTOM → Symptom`, `Patient → HAS_MEASUREMENT → Measurement`, `Disease → HAS_RISK_FACTOR → RiskFactor` và `Recommendation → SUPPORTED_BY → Source`. Đơn vị đo, thời điểm, bối cảnh mẫu và nguồn hướng dẫn cần nằm trong thuộc tính phù hợp. Ghi nhận triệu chứng hoặc phép đo riêng với quan hệ chẩn đoán đã được xác nhận. Graph hiện tại thực hiện phần **kiến thức tham khảo**, với đoạn trích và nguồn để kiểm chứng.

### Câu 24. Neo4j hỗ trợ truy xuất và suy luận bằng đường đi như thế nào?

Neo4j có thể duyệt một mẫu nhiều cạnh để trả về cả kết quả và đường dẫn giải thích. Chẳng hạn, từ `type1` đi ngược cạnh `ABOUT` sang đoạn `p02`, rồi theo `FROM_SOURCE` tới Vinmec:

```cypher
MATCH path = (:Concept {id:'type1'})<-[:ABOUT]-(p:Passage {id:'p02'})
             -[:FROM_SOURCE]->(s:Source)
RETURN path, p.id AS passage_id, s.url AS source_url;
```

Kết quả giúp kiểm tra **vì sao đoạn phù hợp** và **nguồn của nó ở đâu**. Muốn suy diễn thêm quan hệ theo quy tắc, cần mô tả quy tắc bằng truy vấn hoặc mã ứng dụng. [Các truy vấn demo](../knowledge_base/cypher/examples.cypher)

### Câu 25. RAG là gì và vì sao kết hợp với LLM?

Retrieval-Augmented Generation bổ sung một bước truy xuất tri thức trước lúc sinh câu trả lời. Bộ truy xuất chọn các đoạn liên quan từ kho ngoài. LLM nhận câu hỏi và các đoạn đó để tổng hợp câu trả lời. Điều này giúp cập nhật kho dữ liệu độc lập với mô hình và tạo điều kiện đối chiếu nguồn. Chất lượng vẫn phụ thuộc vào đoạn được truy xuất và việc mô hình tuân theo bằng chứng.

Project của em thực hiện bước truy xuất và **trả lời theo hướng trích xuất**. Hệ thống chọn nội dung đã biên soạn trong kho và trả kèm nguồn. Khâu sinh văn bản bằng LLM thuộc kiến trúc RAG tổng quát. [Bài báo RAG của Lewis và cộng sự](https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

### Câu 26. Kiến trúc RAG đầy đủ

**Khi lập chỉ mục**, hệ thống tiếp nhận web, PDF hoặc văn bản, làm sạch, chia thành đoạn, gắn metadata, tính embedding cho từng đoạn và lưu vector để tìm kiếm. **Khi hỏi đáp**, hệ thống xử lý câu hỏi, tính embedding tương thích, truy xuất top-k theo độ tương đồng, có thể xếp hạng lại, rồi đưa các đoạn vào ngữ cảnh của bộ sinh. Cuối cùng, nó trả câu trả lời kèm nguồn hoặc báo thiếu bằng chứng.

Trong project, đoạn được biên soạn sẵn theo từng nguồn JSON rồi nạp vào Neo4j. Truy xuất dùng khái niệm và cạnh đồ thị, sau đó xếp hạng bằng **TF-IDF ký tự** cùng tiêu chí về chủ đề và nguồn. Câu trả lời lấy từ đoạn đứng đầu sau kiểm tra.

### Câu 27. So sánh tìm từ khóa, tìm vector và tìm bằng Knowledge Graph

| Cách truy xuất | Ưu điểm | Giới hạn |
|---|---|---|
| **Từ khóa** như BM25 | Nhanh và dễ hiểu, có lợi khi truy vấn chứa đúng thuật ngữ hoặc mã | Có thể bỏ sót cách nói khác nếu từ vựng ít trùng |
| **Tương đồng vector** | Có thể tìm cách diễn đạt gần nghĩa bằng embedding phù hợp | Phụ thuộc mô hình embedding và có thể trả đoạn cùng chủ đề nhưng sai trọng tâm |
| **Knowledge Graph** | Lần được quan hệ có kiểu, bằng chứng và nguồn | Cần schema, dữ liệu được gắn quan hệ và bước nhận diện thực thể |

Project kết hợp **đồ thị để chọn và giải thích ứng viên** với **TF-IDF thưa để xếp hạng theo mức gần nhau về mặt từ vựng**.

### Câu 28. Neo4j và RAG được tích hợp vào trợ lý tiểu đường thế nào?

`GraphRAG.answer()` chuẩn hóa câu hỏi, đối chiếu các cách gọi để nhận diện khái niệm rồi đưa các ID vào tham số `$anchors` của Cypher. Neo4j trả đoạn ứng viên, quan hệ và URL. Python xếp hạng. Explanation Agent kiểm tra nguồn và trả đoạn đứng đầu cùng bằng chứng. Khi nguồn có nhận định gắn nhãn mâu thuẫn trong cùng phạm vi, hệ thống chuyển sang trạng thái cần xem xét. Với câu hỏi cá nhân hoặc về liều thuốc, Rule Agent định tuyến trước khi truy xuất. [Mã RAG](../core/rag.py) · [Cypher dùng lúc chạy](../knowledge_base/cypher/retrieve.cypher)

## Phần 4. Intelligent Agents và Multi Agent Systems

### Câu 29. Intelligent Agent là gì?

Tác tử nhận quan sát từ **môi trường**, duy trì hoặc suy ra **trạng thái**, quyết định và thực hiện **hành động** để hướng tới **mục tiêu**. Ví dụ, robot quan sát bản đồ và vật cản, xác định vị trí hiện tại, chọn di chuyển và cố tới đích. Trong project, Knowledge Agent nhận câu hỏi đã qua kiểm tra và chủ đề trước đó, rồi thực hiện hành động truy xuất để lấy bằng chứng phù hợp.

### Câu 30. Kiến trúc chung của một tác tử

Tác tử nhận input, xử lý hoặc chuẩn hóa quan sát, tra tri thức hay cập nhật trạng thái, dùng cơ chế quyết định để chọn hành động, rồi xuất kết quả. Cơ chế quyết định có thể là tập luật, mô hình học máy hoặc bộ lập kế hoạch. Phản hồi từ môi trường có thể được dùng ở bước sau. Ví dụ, Rule Agent nhận câu hỏi, chuẩn hóa văn bản, đối chiếu quy tắc và trả một thông điệp có trạng thái như cho phép truy xuất hoặc cần người có chuyên môn xem xét.

### Câu 31. Single-agent khác Multi-Agent System thế nào?

Một hệ single-agent giao nhiệm vụ cho một tác tử. MAS chia công việc cho nhiều tác tử có vai trò và kết quả trung gian riêng. MAS phù hợp khi nhiệm vụ có thể phân rã, dùng nhiều công cụ, hoặc cần biết bước nào gây lỗi. Ví dụ, bài toán của em cần kiểm tra hồ sơ, chạy MLP, truy xuất Neo4j và kiểm tra nguồn. Cái giá là thêm giao tiếp, điều phối và khả năng lỗi lan giữa các bước.

### Câu 32. Một MAS được tổ chức như thế nào?

Thiết kế cần mô tả vai trò, quyền đọc hoặc ghi, định dạng thông điệp, cách giao việc, cách kết hợp kết quả và xử lý lỗi. Agent có thể chia sẻ trạng thái trong một kho chung hoặc gửi kết quả cho nhau. Bộ điều phối quyết định điều kiện chạy tuần tự hoặc song song theo quan hệ phụ thuộc. Trong project, Data Agent cung cấp đầu vào hợp lệ trước khi Prediction Agent chạy, còn Rule Agent phải cho phép trước khi Knowledge Agent truy vấn kho.

### Câu 33. Kiến trúc tập trung, phân tán và phân cấp

- **Tập trung.** Một coordinator quyết định luồng. Dễ kiểm soát thứ tự và theo dõi, nhưng coordinator có thể thành nút thắt hoặc điểm lỗi chung.
- **Phân tán.** Tác tử trao đổi và tự phối hợp. Có thể mở rộng và phân chia trách nhiệm, nhưng đồng bộ và giải quyết xung đột khó hơn.
- **Phân cấp.** Một số điều phối viên quản lý các nhóm tác tử. Phù hợp công việc lớn có cây nhiệm vụ, nhưng có thêm tầng xử lý và chi phí phối hợp.

Project dùng dạng **tập trung**, với Coordinator gọi các lớp Python trong cùng ứng dụng.

### Câu 34. Tác tử giao tiếp và giao việc thế nào?

Thông điệp có thể chứa ID yêu cầu, tên tác tử, trạng thái, payload, bằng chứng và phiên bản. Tác tử cũng có thể cùng đọc kho tri thức hoặc dùng trạng thái chung có quy tắc truy cập. Khi giao việc, tác tử điều phối gửi nhiệm vụ, dữ liệu cần thiết và điều kiện hoàn thành, rồi nhận kết quả hoặc lỗi. Trong project, `Message` truyền đầu ra có cấu trúc qua lời gọi Python. Monitoring dùng cùng `request_id` để nối các bước thành một trace. [Cấu trúc Message và Monitoring](../core/agents.py)

### Câu 35. Reinforcement Learning và cách học qua tương tác

Tác tử quan sát trạng thái, chọn hành động theo policy, nhận phần thưởng và chuyển sang trạng thái mới. Mục tiêu thường là tối đa hóa tổng phần thưởng chiết khấu kỳ vọng thay vì chỉ chọn phần thưởng tốt ngay bước hiện tại. Trong huấn luyện, tác tử cân bằng **khám phá** hành động mới với **khai thác** hành động đã biết tốt. Ví dụ, robot trong mô phỏng nhận thưởng khi tới đích và bị trừ điểm khi va chạm, từ đó học đường đi tốt qua nhiều lượt.

### Câu 36. Markov Decision Process gồm những gì?

MDP thường được viết `(S, A, P, R, γ)`. `S` là không gian trạng thái, `A` là tập hành động, `P(s′|s,a)` là xác suất sang trạng thái mới, `R` mô tả phần thưởng và `γ` là hệ số chiết khấu. Policy `π(a|s)` cho biết cách chọn hành động theo trạng thái và là đối tượng cần tìm. Tính Markov nghĩa là khi biết trạng thái hiện tại và hành động, phân bố của trạng thái tiếp theo không cần toàn bộ lịch sử. Ví dụ, trong mê cung tĩnh, vị trí hiện tại có thể đủ để mô tả trạng thái.

### Câu 37. AI agent khác thành phần phần mềm truyền thống ở đâu?

Khác biệt nằm ở cách hệ thống chọn hành động và mức tự chủ. Một thành phần thường thực hiện một thao tác đã định khi được gọi. Tác tử có mục tiêu cục bộ và có thể chọn hành vi, dùng công cụ hoặc trao đổi dựa trên đầu vào và trạng thái. Tác tử có thể dựa trên luật hoặc dùng mô hình học máy. Trong project, tác tử là các thành phần chuyên trách có giao diện rõ ràng. Quyền quyết định luồng tổng thể thuộc Coordinator.

### Câu 38. Các thách thức khi phát triển MAS

Các thách thức chính gồm thứ tự phụ thuộc, định dạng thông điệp không khớp, dữ liệu trùng hoặc cũ, tác tử đưa kết luận xung đột, lỗi dịch vụ ngoài, giới hạn quyền truy cập và tăng độ trễ. Giải pháp cần gắn vào lỗi cụ thể, ví dụ kiểm tra schema, quy định bước được phép chạy tiếp, giới hạn quyền truy vấn và đối chiếu bằng chứng. Đánh giá nên đo cả từng tác tử lẫn lượt chạy đầu cuối. Project có kiểm thử hồ sơ sai, Neo4j không sẵn sàng, xung đột bằng chứng giả lập và trace theo mã yêu cầu.

## Phần 5. Ứng dụng thực hành và Diabetes Hybrid Lab

### Câu 39. Dữ liệu, đặc trưng, nhãn, kiến trúc và đầu ra của mô hình diabetes

Mô hình trên web dùng **BRFSS 2015 đã xử lý**, gồm **253.680 hồ sơ**, với **21 đặc trưng khảo sát**. Ví dụ có BMI, mã nhóm tuổi, huyết áp cao, hoạt động thể chất và sức khỏe tự báo cáo. Nhãn `Diabetes_012` có ba giá trị **0** không báo cáo tiểu đường, **1** báo cáo tiền tiểu đường, **2** báo cáo tiểu đường. Mã lớp 2 không chứa thông tin típ bệnh.

Em dùng PyTorch MLP **21 → 64 → 32 → 3**. Khi chạy, đầu vào được chuẩn hóa theo thống kê đã lưu, mô hình tạo ba logit và softmax chuyển chúng thành ba điểm. Web chọn nhãn có điểm cao nhất, hiển thị cả ba điểm và phiên bản mô hình. Đây là điểm gắn với nhãn khảo sát và chưa được hiệu chuẩn. Web mặc định dùng checkpoint gốc của Assignment 4. [Khoản mục dữ liệu và mô hình trong README](../README.md) · [Mã suy luận](../core/model.py)

### Câu 40. Em phát triển và triển khai ứng dụng dự đoán như thế nào?

Em xác định schema 21 biến, chuẩn bị bảng dữ liệu, lưu bộ chuẩn hóa và checkpoint, rồi đưa suy luận vào phần backend. Khi Flask khởi động, nó nạp các checkpoint. Khi chạy yêu cầu, giao diện gửi chế độ và hồ sơ giả lập. Backend kiểm tra đầu vào, Coordinator gọi Prediction Agent, và `Predictor.predict()` chuẩn hóa rồi chạy mạng ở chế độ suy luận. Web nhận báo cáo có ba điểm, nhãn, phiên bản và trạng thái.

Với bản demo trên máy em, mở instance **Diabetes** trong Neo4j Desktop để phục vụ nhánh tri thức, rồi chạy `python app.py` từ thư mục dự án. Hồ sơ người dùng trong demo là hồ sơ giả lập. [Mã Flask](../app.py) · [Hướng dẫn demo](DEMO_GUIDE_VI.md)

### Câu 41. So sánh ứng dụng tiểu đường, cổ phiếu và giá nhà của em

| Ứng dụng | Đầu vào | Mục tiêu và đầu ra | Chỉ số và kết quả thực tế |
|---|---|---|---|
| **Diabetes Hybrid Lab** | 21 biến khảo sát BRFSS của một hồ sơ | Phân loại một trong ba nhãn khảo sát | Trong thí nghiệm theo nhóm hồ sơ, MLP đạt macro-F1 **0,4237**, Logistic Regression đạt **0,4249**. |
| **Cổ phiếu AAPL, Assignment 6** | Cửa sổ **14 phiên giao dịch** với giá, khối lượng và đặc trưng dẫn xuất | Hồi quy giá đóng cửa ở phiên kế tiếp | RNN PyTorch có RMSE **20,6136**. Baseline dùng giá phiên cuối có RMSE **5,0779**. |
| **Giá nhà HDB Singapore, Assignment 3** | Đặc trưng bảng như loại căn hộ, diện tích, vị trí và thời hạn thuê còn lại | Hồi quy giá bán lại `resale_price` tính bằng SGD | HistGradientBoosting có RMSE **31.489,94 SGD**, thấp nhất trong bảng kết quả đã lưu. |

Trong bài cổ phiếu, thứ tự thời gian và baseline giá gần nhất rất quan trọng. Trong bài nhà, đặc trưng vị trí và thông tin căn hộ được xử lý như dữ liệu bảng. Các giá trị RMSE thuộc từng bài toán và từng đơn vị nên chỉ dùng để so với đối chứng trong cùng bài. [Diabetes BRFSS](../artifacts/experiment.json) · [AAPL](../../../assignment%206/DATA/outputs/metrics/all_experiments_summary.csv) · [HDB](../../../assignment%203/DATA/housePrice/housePrice_model_comparison.csv)

### Câu 42. Demo một mẫu đầu vào mới được xử lý ra sao

1. Trên web chọn **Chỉ chạy MLP**. Chọn **Hồ sơ B**, hoặc bật phần nhập 21 trường và dùng một bộ giá trị giả lập hợp lệ.
2. Chỉ vào dữ liệu gửi từ giao diện. Data Agent kiểm tra tên trường, kiểu số nguyên và phạm vi. Vector sau đó được sắp theo đúng thứ tự mô hình.
3. Prediction Agent gọi `Predictor.predict()`. Hàm chuẩn hóa bằng trung bình và độ lệch chuẩn lúc huấn luyện, chạy MLP rồi áp dụng softmax và chọn nhãn bằng `argmax`.
4. Chỉ đầu ra gồm ba điểm, nhãn khảo sát và tên checkpoint. Mở Trace để thấy thứ tự Data, Prediction, Rule, Explanation và Monitoring.

Trong một lượt kiểm tra với hồ sơ B và checkpoint gốc, ba điểm khoảng **0,170 / 0,303 / 0,528**, tương ứng theo thứ tự lớp 0, 1, 2. Khi vấn đáp, đọc số đang hiển thị trong lượt chạy. [Web demo](http://127.0.0.1:8012/#demo) · [Mã Predictor](../core/model.py)

### Câu 43. Kiến trúc tổng thể của Diabetes Hybrid Lab và lý do dùng MAS

Kiến trúc là **điều phối tập trung**. Flask nhận yêu cầu và chuyển cho Coordinator. Nhánh mô hình gồm xác thực hồ sơ, suy luận và kiểm tra điểm. Nhánh tri thức gồm kiểm tra phạm vi, truy xuất Neo4j và kiểm tra nguồn. Explanation Agent tập hợp đầu ra trong một báo cáo. Monitoring Agent ghi trace cho cả lượt.

Em chọn MAS vì từng công việc có đầu vào và đầu ra rõ ràng. Ví dụ, lỗi kết nối Neo4j có thể được ghi ở nhánh tri thức trong khi mô hình cục bộ vẫn trả điểm. Người xem có thể theo dõi riêng bước nào thực hiện và bước nào lỗi. Đổi lại, thiết kế cần giao diện thông điệp và coordinator để quản lý thứ tự. [Sơ đồ luồng](../artifacts/figures/01_agent_flow.png) · [Sơ đồ tuần tự](../artifacts/figures/09_sequence_diagram.png)

### Câu 44. Liệt kê và giải thích bảy tác tử

| Tác tử | Nhận vào | Trả về hoặc tương tác |
|---|---|---|
| **Coordinator** | Chế độ, hồ sơ, câu hỏi và ngữ cảnh nếu có | Tạo mã yêu cầu, gọi các tác tử và trả báo cáo |
| **Data Agent** | Hồ sơ giả lập | Vector 21 trường theo đúng thứ tự hoặc lỗi xác thực |
| **Prediction Agent** | Vector hợp lệ và checkpoint | Ba điểm, nhãn khảo sát và phiên bản. Rule Agent dùng các điểm để xem độ chênh. |
| **Rule Agent** | Câu hỏi hoặc kết quả mô hình | Trạng thái cho phép truy xuất, hướng dẫn tiếp theo hoặc cờ cần xem xét |
| **Knowledge Agent** | Câu hỏi đã qua kiểm tra và ngữ cảnh | Gọi RAG và Neo4j để lấy đoạn trích, nguồn, hoặc báo kho chưa sẵn sàng |
| **Explanation Agent** | Kết quả mô hình, luật và bằng chứng | Kiểm tra nguồn và tạo báo cáo có điểm, phần tri thức, trạng thái cần xem xét và giới hạn |
| **Monitoring Agent** | Thông điệp tại từng bước | Trace với mã yêu cầu, trạng thái, thời gian, phiên bản và mã bằng chứng |

[Mã của bảy tác tử](../core/agents.py)

### Câu 45. Mô hình được tích hợp và truyền kết quả giữa các tác tử ra sao?

Ở chế độ dự đoán hoặc kết hợp, Coordinator gọi Data Agent. Nếu dữ liệu hợp lệ, nó truyền thông điệp chứa vector đặc trưng cho **Prediction Agent**. Prediction Agent gọi `Predictor.predict()` trên checkpoint đã chọn. Kết quả đi vào một `Message` cùng `request_id`, trạng thái, điểm, nhãn và phiên bản. Rule Agent đọc hai điểm cao nhất và tạo cờ `review_required` khi độ chênh nhỏ hơn **0,10** theo quy tắc demo. Explanation Agent nhận cả điểm và kết quả quy tắc để ghi báo cáo. [Mã điều phối](../core/agents.py)

### Câu 46. Em dùng Neo4j và Knowledge Graph ở đâu?

Trình diễn ba nhãn **Concept**, **Passage** và **Source**. Một passage nối chủ đề qua `ABOUT` và nơi xuất bản qua `FROM_SOURCE`. Giữa các concept còn có quan hệ ngữ nghĩa như `CLASSIFIED_WITH`. Builder đọc **22 tệp nguồn** và tạo **21 concept, 27 passage** trong bộ dữ liệu của project. Tổng số nút mà Desktop hiển thị có thể lớn hơn nếu database chứa dữ liệu đã nạp trước đó.

Trong Neo4j Desktop, chọn database đã nạp kho rồi chạy truy vấn minh họa ở **câu 24**. Mở kết quả đồ thị để chỉ ba nút và mở dạng bảng để đọc URL Vinmec. Sau đó mở truy vấn thực thi để chỉ biến `$anchors` mà Python truyền từ câu hỏi. [Tệp nạp Neo4j](../knowledge_base/cypher/kb_diabet.cypher) · [Ví dụ demo](../knowledge_base/cypher/examples.cypher) · [Truy vấn thực thi](../knowledge_base/cypher/retrieve.cypher)

### Câu 47. Em tích hợp RAG và giảm câu trả lời thiếu căn cứ như thế nào?

Rule Agent kiểm tra phạm vi trước. `GraphRAG.answer()` chuẩn hóa câu hỏi, tìm khái niệm theo các tên tương đương, truy vấn Neo4j bằng Cypher tham số hóa, rồi xếp hạng theo TF-IDF ký tự, khớp khái niệm và nhóm nguồn. Explanation Agent kiểm tra URL nguồn và chọn đoạn đứng đầu để hiển thị. Các đoạn còn lại xuất hiện trong **Nguồn và bằng chứng**. Cách trích xuất giúp đối chiếu câu trả lời trực tiếp với đoạn lưu trong Neo4j.

Hệ thống có thể trả trạng thái thiếu bằng chứng. Nếu các đoạn đã gắn nhãn chứa nhận định mâu thuẫn trong cùng phạm vi, hệ thống giữ danh sách nguồn và yêu cầu xem xét. Chất lượng vẫn phụ thuộc độ bao phủ và cách xếp hạng. Câu hỏi quá cụ thể đôi khi nhận đoạn cùng chủ đề nhưng chưa sát ý. [Mã retrieval](../core/rag.py) · [Mã kiểm tra và tổng hợp](../core/agents.py)

### Câu 48. Luồng thực thi khi người dùng hỏi về tiểu đường

Với **Chỉ hỏi đáp**, Flask nhận JSON rồi gọi Coordinator. Coordinator tạo mã yêu cầu và gọi Rule Agent. Nếu câu hỏi đủ điều kiện, Knowledge Agent truy xuất Neo4j. Explanation Agent kiểm tra nguồn và tạo báo cáo. Monitoring trả trace cùng mã yêu cầu. Prediction Agent chỉ được gọi khi người dùng chọn chế độ có chạy mô hình.

Với **Chạy cả hai nhánh**, Coordinator xử lý Data → Prediction → Rule kiểm tra điểm trước. Nếu có câu hỏi, nó tiếp tục Rule kiểm tra phạm vi → Knowledge khi được phép → Explanation. Điểm BRFSS và bằng chứng văn bản xuất hiện ở hai trường riêng trong báo cáo. Nếu hồ sơ đầu vào sai, Coordinator báo lỗi trước khi chạy các bước đó.

### Câu 49. Kịch bản demo hoàn chỉnh

1. Trong Neo4j Desktop bật instance **Diabetes**. Từ terminal ở thư mục chứa `app.py`, chạy `python app.py`. Kiểm tra [trạng thái kho](http://127.0.0.1:8012/api/graph) có `ready: true`.
2. Mở Query trong Desktop và chạy truy vấn ở câu 24 để chỉ `p02`, `type1` và URL Vinmec.
3. Mở web, chọn **Hồ sơ B**, giữ **MLP gốc từ Assignment 4**, nhập **“Chẩn đoán tiểu đường type 1 bằng xét nghiệm nào?”** và bấm **Chạy cả hai nhánh**.
4. Chỉ ba điểm khảo sát, mở **Nguồn và bằng chứng** để đối chiếu đoạn Vinmec trong kết quả, rồi cuộn tới **Trace tác tử**. Mô tả Data Agent, Prediction Agent, hai bước Rule, Knowledge Agent, Explanation và Monitoring theo đúng thứ tự.
5. Nếu thầy hỏi về chuyển hướng, thử câu xin chỉ định liều. Chỉ trạng thái trong báo cáo và giải thích vì sao nhánh truy xuất giáo dục không được gọi cho tình huống đó.

[Web demo](http://127.0.0.1:8012/#demo) · [Kịch bản chi tiết](DEMO_GUIDE_VI.md) · [Ảnh lượt chạy kết hợp](../demo/02_combined_vinmec.png)

### Câu 50. Em đánh giá hệ đa tác tử như thế nào?

- **Mô hình.** Đánh giá trên tập test giữ riêng. Với BRFSS chia theo nhóm, báo cáo accuracy, macro-F1, ROC-AUC và recall từng lớp. Logistic Regression đạt macro-F1 **0,4249** và MLP đạt **0,4237**. Recall lớp tiền tiểu đường của hai mô hình lần lượt là **0,3253** và **0,2395**.
- **Truy xuất và câu trả lời.** Có **39 ca hỏi đáp nội bộ**. Trong 30 ca có đoạn tham chiếu, **25/30** có đoạn đó ở hạng đầu và **30/30** có trong top 3. Các kiểm tra còn xem câu trả lời trích xuất có khớp đoạn đứng đầu và có URL được chấp nhận hay không. Chưa có bộ chấm độc lập của chuyên gia lâm sàng.
- **Điều phối và độ tin cậy.** Kiểm tra hồ sơ sai, bỏ trống câu hỏi, lựa chọn checkpoint, mất kết nối Neo4j và mâu thuẫn bằng chứng giả lập. Trong phép thử 18 lượt với kết quả kỳ vọng đã định, luồng có Rule Agent đủ bước đạt **18/18**, còn truy xuất trực tiếp đạt **9/18**. Thí nghiệm này đo đúng trạng thái mong đợi trên các tình huống đã thiết kế.
- **Độ trễ.** Monitoring ghi thời gian từng bước và toàn lượt. Có thể mở tệp kết quả để đọc median và p95 của lần chạy đã lưu. Các lần hỏi đáp trên máy cục bộ có thể chênh nhau tùy trạng thái kết nối, tải máy và nhánh xử lý.

[Chỉ số mô hình](../artifacts/experiment.json) · [Kết quả truy xuất và điều phối](../artifacts/demo_verification.json)

### Câu 51. Giới hạn, rủi ro và hướng cải thiện

**Dữ liệu và mô hình.** BRFSS là dữ liệu khảo sát cắt ngang với nhãn tự báo cáo, có mất cân bằng lớp và giới hạn đại diện cho các quần thể khác. Điểm softmax chưa hiệu chuẩn. Nếu phát triển tiếp, cần đánh giá hiệu năng theo từng nhóm, hiệu chuẩn trên dữ liệu riêng và xác thực trên bộ dữ liệu ngoài phù hợp với mục đích sử dụng.

**Tri thức và câu trả lời.** Kho tuyển chọn còn nhỏ. Khớp tên tương đương và TF-IDF có thể phản hồi quá chung với cách diễn đạt lạ hoặc câu có nhiều ý. Cần bổ sung câu hỏi kiểm tra độc lập, rà soát ngày và nguồn của từng đoạn, đánh giá chất lượng câu trả lời với người có chuyên môn và cải thiện cách tổng hợp nhiều đoạn.

**Kiến trúc, an toàn và triển khai.** Khi mở rộng, cần kiểm soát quyền đọc/ghi Neo4j, giữ thông tin xác thực ngoài mã nguồn, lọc dữ liệu nhạy cảm khỏi log, dùng timeout phù hợp và đo tải trước khi triển khai. Các quy tắc hiện tại là quy tắc demo, còn nhận diện mâu thuẫn phụ thuộc đoạn đã gắn nhãn. Dự án hiện phục vụ học tập và trình diễn. Muốn dùng trong quy trình lâm sàng phải có xác thực, giám sát và quy trình chuyển người có chuyên môn phù hợp.
