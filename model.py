# model.py
import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

class SleepStageClassifier(nn.Module):
    """간단한 LSTM 기반의 수면 단계 분류 모델"""
    def __init__(self, input_dim=3000, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.5):
        super(SleepStageClassifier, self).__init__()
        
        self.input_dim = input_dim # 30초 * 100Hz
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM 레이어: 시퀀스 데이터의 시간적 특징을 학습합니다.
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,  # 입력 텐서의 차원을 (batch, seq, feature)로 설정
            dropout=dropout,
            bidirectional=True # 양방향 LSTM
        )
        
        # 분류를 위한 Fully Connected 레이어
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # 양방향이므로 *2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, channels, samples_per_epoch)
        batch_size, seq_len, channels, samples = x.shape
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, features)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # Gather last valid hidden state for each sequence
            last_indices = (lengths - 1).clamp(min=0).to(lstm_out.device)
            last_indices = last_indices.view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
            last_hidden_state = lstm_out.gather(1, last_indices).squeeze(1)
        else:
            lstm_out, _ = self.lstm(x)
            last_hidden_state = lstm_out[:, -1, :]

        out = self.dropout(last_hidden_state)
        out = self.fc(out)
        return out


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds maximum positional encoding length {self.pe.size(1)}."
            )
        return x + self.pe[:, : x.size(1), :]


class AdditiveAttention(nn.Module):
    """Additive attention similar to the TensorFlow implementation used in SleepTransformer."""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (batch, seq_len, dim)
        attn = torch.tanh(self.proj(x))  # (batch, seq_len, attention_dim)
        scores = self.score(attn).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class SleepTransformer(nn.Module):
    """PyTorch adaptation of the SleepTransformer architecture."""

    def __init__(
        self,
        num_channels: int,
        samples_per_epoch: int,
        seq_len: int,
        num_classes: int = 5,
        frame_size: int = 100,
        frame_stride: int = None,
        frame_embed_dim: int = None,
        frame_num_layers: int = 2,
        frame_num_heads: int = 8,
        frame_ff_dim: int = 1024,
        seq_embed_dim: int = None,
        seq_num_layers: int = 2,
        seq_num_heads: int = 8,
        seq_ff_dim: int = 1024,
        attention_dim: int = 64,
        fc_hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.samples_per_epoch = samples_per_epoch
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.frame_size = frame_size
        self.frame_stride = frame_stride or frame_size
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")
        if self.frame_size > samples_per_epoch:
            raise ValueError("frame_size must be <= samples_per_epoch.")

        self.num_frames = 1 + (samples_per_epoch - frame_size) // self.frame_stride
        if self.num_frames <= 0:
            raise ValueError("Computed zero frames per epoch. Check frame_size and frame_stride.")

        flattened_frame_dim = num_channels * frame_size
        frame_embed_dim = frame_embed_dim or flattened_frame_dim
        if frame_embed_dim % frame_num_heads != 0:
            frame_embed_dim = frame_num_heads * math.ceil(frame_embed_dim / frame_num_heads)
        seq_embed_dim = seq_embed_dim or frame_embed_dim
        if seq_embed_dim % seq_num_heads != 0:
            seq_embed_dim = seq_num_heads * math.ceil(seq_embed_dim / seq_num_heads)

        self.frame_proj = nn.Linear(flattened_frame_dim, frame_embed_dim)
        self.frame_pos_enc = PositionalEncoding(frame_embed_dim, max_len=self.num_frames)
        frame_encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_embed_dim,
            nhead=frame_num_heads,
            dim_feedforward=frame_ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.frame_encoder = nn.TransformerEncoder(frame_encoder_layer, num_layers=frame_num_layers)
        self.frame_attention = AdditiveAttention(frame_embed_dim, attention_dim)

        if frame_embed_dim != seq_embed_dim:
            self.seq_input_proj = nn.Linear(frame_embed_dim, seq_embed_dim)
        else:
            self.seq_input_proj = nn.Identity()

        self.seq_pos_enc = PositionalEncoding(seq_embed_dim, max_len=seq_len)
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_embed_dim,
            nhead=seq_num_heads,
            dim_feedforward=seq_ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(seq_encoder_layer, num_layers=seq_num_layers)
        self.seq_attention = AdditiveAttention(seq_embed_dim, attention_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(seq_embed_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.out = nn.Linear(fc_hidden_dim, num_classes)

    def _epoch_to_frames(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels, samples_per_epoch)
        batch, seq_len, channels, samples = x.shape
        if samples != self.samples_per_epoch:
            raise ValueError(
                f"Expected {self.samples_per_epoch} samples per epoch, but got {samples}."
            )
        x = x.view(batch * seq_len, channels, samples)
        frames = x.unfold(dimension=2, size=self.frame_size, step=self.frame_stride)
        # frames: (batch*seq_len, channels, num_frames, frame_size)
        frames = frames.permute(0, 2, 1, 3).contiguous()
        frames = frames.view(batch * seq_len, frames.size(1), -1)  # (batch*seq_len, num_frames, channels*frame_size)
        return frames

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, _, _ = x.shape
        frames = self._epoch_to_frames(x)  # (batch*seq_len, num_frames, flattened_dim)
        frames = self.frame_proj(frames)
        frames = self.frame_pos_enc(frames)
        frames = self.frame_encoder(frames)
        frame_summary, _ = self.frame_attention(frames)  # (batch*seq_len, frame_embed_dim)
        frame_summary = frame_summary.view(batch, seq_len, -1)

        seq_input = self.seq_input_proj(frame_summary)
        seq_input = self.seq_pos_enc(seq_input)
        seq_output = self.seq_encoder(seq_input)

        if lengths is not None:
            lengths = lengths.clamp(min=1)
            mask = torch.arange(seq_output.size(1), device=seq_output.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None

        seq_summary, _ = self.seq_attention(seq_output, mask=mask)
        z = self.dropout(F.gelu(self.fc1(seq_summary)))
        z = self.dropout(F.gelu(self.fc2(z)))
        logits = self.out(z)
        return logits


class DeepSleepNet(nn.Module):
    """
    TensorFlow로 구현된 DeepSleepNet을 PyTorch 스타일로 변환한 모델.
    
    SleepStageClassifier와 완벽하게 동일한 입출력 규격을 따르도록 수정되었습니다.
    다중 채널 입력과 가변 길이 시퀀스(lengths)를 모두 지원합니다.
    """
    def __init__(self, 
                 num_channels=1,
                 samples_per_epoch=3000, 
                 n_classes=5, 
                 seq_length=25, 
                 n_rnn_layers=2, 
                 use_dropout=True):
        super(DeepSleepNet, self).__init__()

        self.num_channels = num_channels
        self.samples_per_epoch = samples_per_epoch
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.use_dropout = use_dropout
        
        # --- 특징 추출기 (Feature Extractor) ---
        self.feature_extractor_small = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=50, stride=6, padding=22),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5 if use_dropout else 0.0),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten()
        )

        self.feature_extractor_large = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=400, stride=50, padding=175),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.5 if use_dropout else 0.0),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        feature_size_small = self._get_conv_output_size(self.feature_extractor_small)
        feature_size_large = self._get_conv_output_size(self.feature_extractor_large)
        self.cnn_output_size = feature_size_small + feature_size_large
        print(f"Flattened CNN feature size: {self.cnn_output_size}")

        self.feature_dropout = nn.Dropout(p=0.5 if use_dropout else 0.0)

        # --- 시퀀스 모델 (Sequence Model) ---
        self.lstm_hidden_size = 512
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(0.5 if use_dropout and n_rnn_layers > 1 else 0.0)
        )
        
        lstm_output_size = self.lstm_hidden_size * 2

        # --- 잔차 연결 및 분류기 ---
        self.residual_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, lstm_output_size),
            nn.BatchNorm1d(lstm_output_size),
            nn.ReLU()
        )
        
        self.final_dropout = nn.Dropout(p=0.5 if use_dropout else 0.0)
        self.classifier = nn.Linear(lstm_output_size, n_classes)

    def _get_conv_output_size(self, model):
        dummy_input = torch.randn(1, self.num_channels, self.samples_per_epoch)
        output = model(dummy_input)
        return output.shape[1]

    def forward(self, x, lengths=None):
        batch_size, seq_len, num_chans, num_samples = x.shape
        
        # 1. CNN 특징 추출
        x_reshaped = x.view(-1, num_chans, num_samples)
        features_small = self.feature_extractor_small(x_reshaped)
        features_large = self.feature_extractor_large(x_reshaped)
        features = torch.cat((features_small, features_large), dim=1)
        features = self.feature_dropout(features)
        
        # 2. LSTM을 위한 시퀀스 데이터 생성
        features_seq = features.view(batch_size, seq_len, -1)
        
        # 3. LSTM 처리 및 잔차 연결 (lengths 유무에 따라 분기)
        if lengths is not None:
            # `lengths`가 있는 경우: 패딩을 고려하여 '진짜' 마지막 스텝을 추출
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features_seq, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

            # 각 시퀀스의 마지막 유효한 인덱스 계산
            last_indices = (lengths - 1).clamp(min=0).to(lstm_out.device)
            
            # LSTM 출력에서 마지막 스텝 추출
            last_indices_lstm = last_indices.view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
            last_lstm_out = lstm_out.gather(1, last_indices_lstm).squeeze(1)

            # 잔차 연결을 위한 CNN 특징에서도 마지막 스텝 추출
            last_indices_feat = last_indices.view(-1, 1, 1).expand(-1, 1, features_seq.size(2))
            last_features = features_seq.gather(1, last_indices_feat).squeeze(1)
            
        else:
            # `lengths`가 없는 경우: 모든 시퀀스 길이가 같으므로 맨 마지막(-1) 스텝을 사용
            lstm_out, _ = self.lstm(features_seq)
            last_lstm_out = lstm_out[:, -1, :]
            last_features = features_seq[:, -1, :]

        # 4. 잔차 연결 및 최종 분류
        residual = self.residual_fc(last_features)
        combined = last_lstm_out + residual
        combined = self.final_dropout(combined)
        logits = self.classifier(combined)
        
        return logits


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "lstm": SleepStageClassifier,
    "sleep_transformer": SleepTransformer,
    "deepsleepnet": DeepSleepNet,
}


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[key]
    return model_cls(**kwargs)
