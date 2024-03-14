import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from google.colab import drive

drive.mount('/content/drive', force_remount=True)

class TextTransform:
    def __init__(self):
        self.char_map = {'а': 0, 'б': 1, 'в': 2, 'г': 3, 'д': 4, 'е': 5, 'ё': 6, 'ж': 7, 'з': 8, 'и': 9,
                         'й': 10, 'к': 11, 'л': 12, 'м': 13, 'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18,
                         'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25, 'щ': 26, 'ъ': 27,
                         'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, ' ': 33}

    def text_to_int(self, text):
        return [self.char_map[char] for char in text]

    def int_to_text(self, labels):
        return ''.join([char for label in labels for char, idx in self.char_map.items() if idx == label])

class SpeechDataset(Dataset):
    MAX_INPUT_LENGTH = 160

    def __init__(self, file_paths, text_transform):
        self.file_paths = file_paths
        self.text_transform = text_transform
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=800, hop_length=160, n_mels=160
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        if not os.path.exists(file_path):
            print(f"Error: File does not exist - {file_path}")
            return None

        try:
            waveform, _ = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        feature = self.mel_spectrogram_transform(waveform)

        if feature.shape[-1] > self.MAX_INPUT_LENGTH:
            feature = feature[..., :self.MAX_INPUT_LENGTH]
        elif feature.shape[-1] < self.MAX_INPUT_LENGTH:
            padding = torch.zeros((1, feature.shape[1], self.MAX_INPUT_LENGTH - feature.shape[-1]))
            feature = torch.cat((feature, padding), dim=-1)

        return feature, file_path

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    features = [item[0] for item in batch]
    file_paths = [item[1] for item in batch]

    features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return features, file_paths

class SpeechToTextRCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechToTextRCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x, _ = self.lstm(x.transpose(1, 2))
        return F.log_softmax(self.fc(x), dim=2)

def decode_predictions(outputs, text_transform):
    _, preds = outputs.max(2)
    preds = preds.transpose(0, 1).contiguous()
    decoded_preds = []
    for pred in preds:
        decoded_pred = text_transform.int_to_text(pred)
        decoded_preds.append(decoded_pred)
    return decoded_preds

def load_model(model_path, device, input_size, hidden_size, output_size):
    model = SpeechToTextRCNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)

DATASET_PATH = '/content/drive/MyDrive/y'
MODEL_SAVE_PATH = '/content/drive/MyDrive/m.pt'
BATCH_SIZE = 64
NUM_MELS = 160
EMBEDDING_SIZE = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_transform = TextTransform()
file_paths = ['/content/drive/MyDrive/y/a.wav']

val_dataset = SpeechDataset(file_paths=file_paths, text_transform=text_transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = load_model(MODEL_SAVE_PATH, device, NUM_MELS, EMBEDDING_SIZE, len(text_transform.char_map) + 1)
infer(model, val_dataloader, text_transform, device)