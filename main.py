import argparse
import random
import warnings
import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocess import DEFAULT_FEATURES, get_dataloader
from model import build_model


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train(model, dataloader, criterion, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for batch in dataloader:
        if len(batch) == 3:
            inputs, lengths, labels = batch
        else:
            inputs, labels = batch
            lengths = None

        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device) if lengths is not None else None

        optimizer.zero_grad()
        outputs = model(inputs, lengths=lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, lengths, labels = batch
            else:
                inputs, labels = batch
                lengths = None

            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device) if lengths is not None else None

            outputs = model(inputs, lengths=lengths)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)

    dataset_name = args.data.lower()
    if dataset_name == "capslpdb":
        default_dir = "/data/ceragem/physionet.org/files/capslpdb/1.0.0"
        data_dir = args.data_dir or default_dir
        pattern = os.path.join(data_dir, "*.edf")
        data_format = "capslpdb"
        default_features = list(DEFAULT_FEATURES["capslpdb"])
    elif dataset_name == "sleepbrl":
        default_dir = "/data/ceragem/physionet.org/files/sleepbrl/1.0.0"
        data_dir = args.data_dir or default_dir
        pattern = os.path.join(data_dir, "*.edf")
        data_format = "sleepbrl"
        default_features = list(DEFAULT_FEATURES["sleepbrl"])
    else:
        default_dir = os.path.join(
            "/data/ceragem/physionet.org/files", args.data, "2.1.0", "data_100Hz"
        )
        data_dir = args.data_dir or default_dir
        pattern = os.path.join(data_dir, "*.csv")
        data_format = "dreamt"
        default_features = list(DEFAULT_FEATURES["dreamt"])

    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No data files found under {data_dir} (pattern: {pattern}).")

    files = all_files[:]
    rnd = random.Random(args.seed)
    rnd.shuffle(files)
    split_index = max(1, int(len(files) * (1 - args.test_ratio)))
    train_files = files[:split_index]
    eval_files = files[split_index:]
    if len(eval_files) == 0:
        eval_files = [train_files.pop()]

    print(f"Train files: {len(train_files)} | Eval files: {len(eval_files)}")

    effective_sample_rate = args.sample_rate
    if dataset_name == "sleepbrl" and args.sample_rate == 100:
        effective_sample_rate = 50
        print("Adjusting sample rate to 50 Hz for SleepBRL.")

    requested_features = [c.strip() for c in args.features.split(",") if c.strip()]
    if requested_features:
        feature_columns = requested_features
    elif default_features:
        feature_columns = default_features
    else:
        feature_columns = None

    if feature_columns:
        print(f"Using feature columns: {feature_columns}")
    else:
        print("Using dataset default feature columns.")

    train_loader = get_dataloader(train_files,
                                  args.batch_size,
                                  args.seq_len,
                                  shuffle=True,
                                  epoch_seconds=args.epoch_seconds,
                                  sample_rate_hz=effective_sample_rate,
                                #   max_files=args.max_files,
                                  max_files=None,
                                  
                                  feature_columns=feature_columns,
                                  data_format=data_format)
    eval_loader = get_dataloader(eval_files,
                                 args.batch_size,
                                 args.seq_len,
                                 shuffle=False,
                                 epoch_seconds=args.epoch_seconds,
                                 sample_rate_hz=effective_sample_rate,
                                #  max_files=args.max_files,
                                max_files=None,
                                 feature_columns=feature_columns,
                                 data_format=data_format)

    train_dataset = train_loader.dataset
    num_channels = getattr(train_dataset, "num_channels", train_dataset.sequences.shape[2])
    samples_per_epoch = getattr(train_dataset, "samples_per_epoch", effective_sample_rate * args.epoch_seconds)
    num_classes = 5
    input_dim = getattr(train_dataset, "feature_dim", num_channels * samples_per_epoch)

    print(f"Using {num_channels} channels with {samples_per_epoch} samples per epoch.")

    if args.model.lower() == "lstm":
        model_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    elif args.model.lower() == "sleep_transformer":
        frame_stride = args.frame_stride if args.frame_stride > 0 else None
        model_kwargs = dict(
            num_channels=num_channels,
            samples_per_epoch=samples_per_epoch,
            seq_len=args.seq_len,
            num_classes=num_classes,
            frame_size=args.frame_size,
            frame_stride=frame_stride,
            frame_num_layers=args.frame_layers,
            frame_num_heads=args.frame_heads,
            frame_ff_dim=args.frame_ff_dim,
            seq_num_layers=args.seq_layers,
            seq_num_heads=args.seq_heads,
            seq_ff_dim=args.seq_ff_dim,
            attention_dim=args.attention_dim,
            fc_hidden_dim=args.fc_hidden_dim,
            dropout=args.transformer_dropout,
        )
    elif args.model.lower() == "deepsleepnet":
        model_kwargs = dict(
            num_channels=num_channels,
            samples_per_epoch=samples_per_epoch,
            n_classes=num_classes,
            seq_length=args.seq_len,
            n_rnn_layers=args.num_layers,      # LSTM의 num_layers 인자 재활용
            use_dropout=(args.dropout > 0.0)   # dropout > 0 이면 True로 설정
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = build_model(args.model, **model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []
    start_time = time.perf_counter()
    for epoch in tqdm(range(args.epochs), desc='ceragem'):
        train_loss, train_acc = train(model=model, 
                                      dataloader=train_loader,
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      device=device)
        eval_loss, eval_acc = evaluate(model=model, 
                                       dataloader=eval_loader, 
                                       criterion=criterion, 
                                       device=device)
        losses.append([train_loss, eval_loss]); accs.append([train_acc, eval_acc])
        # Pretty per-epoch logging for clarity
        print(f"Epoch {epoch+1:>3}/{args.epochs}: "
              f"train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | "
              f"train_acc={train_acc:.4f} | eval_acc={eval_acc:.4f}")

    # Summary table
    print("\nSummary (loss: train/eval, acc: train/eval):")
    for i, ((tr_l, ev_l), (tr_a, ev_a)) in enumerate(zip(losses, accs), start=1):
        print(f"  - Epoch {i:>3}: loss {tr_l:.4f}/{ev_l:.4f} | acc {tr_a:.4f}/{ev_a:.4f}")
    elapsed = time.perf_counter() - start_time
    print(f"\nElapsed training time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description='ceragem')

    data_group = parser.add_argument_group("Data loading")
    data_group.add_argument('--data', type=str, default="dreamt")
    data_group.add_argument('--data_dir', type=str, default='', help='Override data directory path')
    data_group.add_argument('--features', type=str, default='', help='Comma-separated feature column list to override defaults')
    data_group.add_argument('--sample_rate', type=int, default=100, help='Samples per second in the raw data')
    data_group.add_argument('--epoch_seconds', type=int, default=30, help='Epoch length in seconds for windowing')
    data_group.add_argument('--seq_len', type=int, default=10)
    data_group.add_argument('--test_ratio', type=float, default=0.1, help='Proportion of subjects for eval set')
    data_group.add_argument('--max_files', type=int, default=None, help='Limit how many files to load per split (useful for quick experiments)')


    train_group = parser.add_argument_group("Training")
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--epochs', type=int, default=10)
    train_group.add_argument('--lr', type=float, default=0.01)
    train_group.add_argument('--seed', type=int, default=0)
    train_group.add_argument('--device', type=int, default=0)

    model_group = parser.add_argument_group("Model (LSTM/Transformer shared)")
    model_group.add_argument('--model', type=str, default='lstm', choices=['lstm', 'sleep_transformer', 'deepsleepnet'])
    model_group.add_argument('--dropout', type=float, default=0.5, help='Dropout for the LSTM model')
    model_group.add_argument('--num_layers', type=int, default=2)
    model_group.add_argument('--hidden_dim', type=int, default=32)

    transformer_group = parser.add_argument_group("Transformer-specific")
    transformer_group.add_argument('--frame_size', type=int, default=100, help='Frame size (samples) for SleepTransformer input windows')
    transformer_group.add_argument('--frame_stride', type=int, default=0, help='Frame stride (samples); 0 defaults to frame_size')
    transformer_group.add_argument('--frame_layers', type=int, default=2, help='Number of frame-level Transformer layers')
    transformer_group.add_argument('--frame_heads', type=int, default=8, help='Number of attention heads in frame-level Transformer')
    transformer_group.add_argument('--frame_ff_dim', type=int, default=1024, help='Feed-forward dimension in frame-level Transformer')
    transformer_group.add_argument('--seq_layers', type=int, default=2, help='Number of sequence-level Transformer layers')
    transformer_group.add_argument('--seq_heads', type=int, default=8, help='Number of attention heads in sequence-level Transformer')
    transformer_group.add_argument('--seq_ff_dim', type=int, default=1024, help='Feed-forward dimension in sequence-level Transformer')
    transformer_group.add_argument('--attention_dim', type=int, default=64, help='Additive attention hidden dimension')
    transformer_group.add_argument('--fc_hidden_dim', type=int, default=1024, help='Hidden dimension of the Transformer classifier head')
    transformer_group.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout used inside Transformer layers')

    args = parser.parse_args()

    main(args)
