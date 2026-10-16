# A7 inference module generated from notebook; preprocessing schema v1.
from pathlib import Path
import json, re, html, unicodedata
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# A7_EXPORT: preprocessing
# Purpose: define pure preprocessing functions shared by the notebook and future web app.
def tokenize(text):
    """Normalize an English review body and return tokens; preserve negation, apostrophes, and !?."""
    text = unicodedata.normalize('NFKC', html.unescape(str(text))).lower()
    text = re.sub(r'<[^>]+>', ' ', text).replace('’', "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[!?]", text)


def encode_texts(texts, vocabulary, max_len):
    """Map tokens to IDs, truncate the tail, pad on the right, and return IDs with the true truncated length."""
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for row, text in enumerate(texts):
        tokens = tokenize(text)[:max_len]
        encoded = [vocabulary.get(token, 1) for token in tokens] or [1]
        ids[row, :len(encoded)] = encoded
        lengths[row] = len(encoded)
    return ids, lengths


# A7_EXPORT: stock_features
# Purpose: calculate features consistently during training and when the web app receives new OHLCV data.
def engineer_stock(frame):
    """Create features using information available through each session; input must include date and OHLCV."""
    frame = frame.copy().sort_values('date').reset_index(drop=True)
    if frame['date'].duplicated().any(): raise ValueError('Duplicate trading dates.')
    frame['log_volume'] = np.log1p(frame['volume'])
    frame['return_1d'] = frame['close'].pct_change(fill_method=None)
    frame['range_pct'] = (frame['high']-frame['low'])/frame['close']
    frame['body_pct'] = (frame['close']-frame['open'])/frame['open']
    frame['ma5_gap'] = frame['close']/frame['close'].rolling(5).mean()-1
    frame['ma20_gap'] = frame['close']/frame['close'].rolling(20).mean()-1
    frame['volatility20'] = frame['return_1d'].rolling(20).std(ddof=1)
    features = ['open','high','low','close','log_volume','return_1d',
                'range_pct','body_pct','ma5_gap','ma20_gap','volatility20']
    frame = frame.replace([np.inf,-np.inf],np.nan).dropna(subset=features).reset_index(drop=True)
    return frame, features


# A7_EXPORT: torch_model
# Purpose: define an explicitly configured model that can be rebuilt from its state_dict for web inference.
class StackedRecurrent(nn.Module):
    """Five-layer RNN/LSTM; reviews use packed sequences and stock uses complete numeric sequences."""
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.task = config['task']
        self.kind = config['kind']
        H = config['hidden']
        if self.task == 'text':
            self.embedding = nn.Embedding(config['vocab_size'],config['embedding'],padding_idx=0)
            input_dim = config['embedding']
        else:
            self.embedding = None
            input_dim = config['input_dim']
        layer_type = nn.RNN if self.kind=='RNN' else nn.LSTM
        self.core = layer_type(input_dim,H,num_layers=5,batch_first=True,dropout=0.0,bidirectional=False)
        self.dropout = nn.Dropout(config['head_dropout'])
        self.head = nn.Linear(H,3 if self.task=='text' else 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialization aligned with Keras calculations: zero biases and total forget bias = 1."""
        if self.embedding is not None:
            nn.init.uniform_(self.embedding.weight,-.05,.05)
            with torch.no_grad(): self.embedding.weight[0].zero_()
        for name,parameter in self.core.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(parameter)
            elif 'weight_hh' in name: nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
                if self.kind=='LSTM' and 'bias_ih' in name:
                    with torch.no_grad(): parameter[self.config['hidden']:2*self.config['hidden']].fill_(1.)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, lengths=None):
        """Return logits or a standardized target; do not apply softmax before cross-entropy."""
        if self.task=='text':
            sequence = self.embedding(x.long())
            if lengths is None: lengths = (x!=0).sum(dim=1).clamp(min=1)
            sequence = pack_padded_sequence(sequence,lengths.cpu(),batch_first=True,enforce_sorted=False)
        else:
            sequence = x.float()
        _,state = self.core(sequence)
        hidden = state[0] if self.kind=='LSTM' else state
        return self.head(self.dropout(hidden[-1]))


# A7_EXPORT: inference
# Purpose: provide inference APIs independent of notebook variables for a future local Flask app.
def load_bundle(model_dir, key):
    """Load a model and preprocessing artifacts from one run directory; call once when the web app starts."""
    model_dir=Path(model_dir)
    registry=json.loads((model_dir/'model_registry.json').read_text(encoding='utf-8'))
    entry=registry[key]
    if entry['framework']=='PyTorch':
        checkpoint=torch.load(model_dir/entry['filename'],map_location='cpu',weights_only=True)
        model=StackedRecurrent(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict']); model.eval()
    else:
        import tensorflow as tf
        model=tf.keras.models.load_model(model_dir/entry['filename'],compile=False)
    bundle={'model':model,'entry':entry}
    if entry['task']=='text':
        bundle['preprocess']=json.loads((model_dir/'tokenizer.json').read_text(encoding='utf-8'))
    else: bundle['preprocess']=joblib.load(model_dir/'stock_preprocess.joblib')
    return bundle


def predict_review(bundle, text):
    """Predict sentiment for one English review; return class probabilities, label, token, OOV, and truncation details."""
    if bundle['entry']['task']!='text': raise ValueError('Selected model is not a sentiment model.')
    tokens=tokenize(text)
    if not tokens: raise ValueError('Review is empty after normalization.')
    pre=bundle['preprocess']
    x,lengths=encode_texts([text],pre['vocab'],pre['max_len'])
    if bundle['entry']['framework']=='PyTorch':
        with torch.no_grad(): logits=bundle['model'](torch.as_tensor(x),torch.as_tensor(lengths)).numpy()[0]
    else: logits=bundle['model'](x,training=False).numpy()[0]
    p=np.exp(logits-logits.max()); p=p/p.sum()
    return {'label':pre['labels'][int(p.argmax())],
            'probabilities':dict(zip(pre['labels'],map(float,p))),
            'confidence':float(p.max()),'tokens_total':len(tokens),'tokens_used':int(lengths[0]),
            'truncated':len(tokens)>pre['max_len'],
            'oov_rate':sum(t not in pre['vocab'] for t in tokens)/len(tokens)}


def predict_next_close(bundle, raw_frame):
    """Forecast one next session from completed OHLCV data; do not fabricate future inputs."""
    if bundle['entry']['task']!='stock': raise ValueError('Selected model is not a stock model.')
    frame=raw_frame.copy()
    frame.columns=frame.columns.str.lower()
    columns=['open','high','low','close','volume']
    if not set(['date']+columns).issubset(frame.columns): raise ValueError('Missing date/OHLCV columns.')
    frame['date']=pd.to_datetime(frame['date'],errors='raise')
    for col in columns: frame[col]=pd.to_numeric(frame[col],errors='raise')
    if not np.isfinite(frame[columns].to_numpy()).all(): raise ValueError('OHLCV contains NaN/Inf values.')
    if (frame[['open','high','low','close']]<=0).any().any() or (frame.volume<0).any():
        raise ValueError('Prices must be positive and volume must be nonnegative.')
    features,names=engineer_stock(frame)
    pre=bundle['preprocess']; L=pre['lookback']
    if names!=pre['features']: raise ValueError('Feature schema does not match training.')
    if len(features)<L: raise ValueError(f'At least {L+20} raw sessions are required for rolling warm-up and lookback.')
    x=pre['x'].transform(features[names].tail(L)).astype('float32')[None]
    if bundle['entry']['framework']=='PyTorch':
        with torch.no_grad(): scaled=bundle['model'](torch.as_tensor(x)).numpy()
    else: scaled=bundle['model'](x,training=False).numpy()
    predicted=float(pre['y'].inverse_transform(scaled)[0,0])
    previous=float(features.close.iloc[-1])
    return {'last_observed_session':features.date.iloc[-1].strftime('%Y-%m-%d'),
            'target':'next trading session','previous_close':previous,'predicted_close':predicted,
            'predicted_change_percent':100*(predicted/previous-1),'input_shape':list(x.shape)}
