# RNN Web Demo - PyTorch Next-Word Prediction

Web demo Flask dành cho notebook `RNN_Next_Word_Prediction_PyTorch.ipynb`.

## 1. Cấu trúc thư mục

```text
assignment root/
├── data/
│   └── wikitext103/
│       └── wikitext-103/
│           ├── wiki.train.tokens
│           ├── wiki.valid.tokens
│           └── wiki.test.tokens
├── models/
│   └── rnn_next_word_best.pt
├── outputs/
│   └── metrics/
│       ├── training_history.json
│       └── test_metrics.json
└── web/
    ├── app.py
    ├── requirements.txt
    ├── export_web_artifacts.py
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

## 2. Cài thư viện

Mở terminal tại thư mục `web/`:

```bash
conda activate CNN
pip install -r requirements.txt
```

## 3. Chạy website

```bash
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

## 4. Model cần có

Web app mặc định tìm:

```text
../models/rnn_next_word_best.pt
```

Đây chính là checkpoint được notebook A5/RNN lưu theo tên `rnn_next_word_best.pt`.

## 5. Training history

Notebook hiện lưu checkpoint model, nhưng để dashboard vẽ training loss, validation loss, perplexity và learning-rate đầy đủ thì cần export `history` và `test_metrics` sau khi train.

Chạy:

```bash
python export_web_artifacts.py
```

Lệnh này in ra code cell để bạn paste vào notebook sau:

```python
history, model = train_model()
test_metrics = evaluate_language_model(model, test_loader, criterion, DEVICE)
```

Sau khi chạy cell export trong notebook, website sẽ tự đọc:

```text
outputs/metrics/training_history.json
outputs/metrics/test_metrics.json
```

## 6. Các phần visualization trong website

- Training loss / validation loss.
- Training perplexity / validation perplexity.
- Learning-rate schedule.
- Top token frequency.
- Parameter composition.
- Dataset split size.
- Next-word probability bar chart.
- Top-k candidate table với probability, log probability và cumulative probability.
- Entropy.
- Top-1 probability.
- Top-5 cumulative probability.
- Top-1 vs Top-2 margin.
- Hidden-state norm.
- Actual context used by RNN.
- Unknown-token count.
- Autoregressive generation speed.
- Per-step next-word probability.
- Per-step hidden-state norm.
- Model architecture.
- Parameter breakdown.
- Dataset token counts, unique counts, file size và UNK rate.

## 7. Lưu ý về tính nhất quán với notebook

Website reconstructs the vocabulary using the same special tokens and frequency ordering used by the notebook:

```text
<unk>
<eos>
most frequent training tokens...
```

Website mặc định cũng dùng `RNN_TRAIN_TOKEN_LIMIT=2,000,000` và `RNN_VOCAB_SIZE=30,000` để khớp với notebook.

Có thể override bằng environment variables:

```bash
set RNN_TRAIN_TOKEN_LIMIT=2000000
set RNN_VOCAB_SIZE=30000
```

Nếu bạn train notebook với vocabulary size hoặc token limit khác, hãy đặt lại các biến trên cho khớp.

## 8. Demo thuyết trình

Một flow phù hợp:

```text
Seed text
   ↓
Context tokens
   ↓
Embedding
   ↓
RNN hidden state
   ↓
Vocabulary logits
   ↓
Temperature
   ↓
Softmax
   ↓
Top-k probabilities
   ↓
Next word
   ↓
Append word
   ↓
Repeat
```

Các mục "Dự đoán từ tiếp theo" và "Sinh văn bản" được thiết kế để thao tác trực tiếp trên sân khấu.
