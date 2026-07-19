import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    audio = [item["audio"] for item in dataset_items]

    padded_audio = pad_sequence(
        audio,
        batch_first=True,
        padding_value=0.0,
    )

    result_batch = {
        "audio": padded_audio,
        "audio_length": torch.tensor(
            [item["audio_length"] for item in dataset_items],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [item["labels"] for item in dataset_items],
            dtype=torch.long,
        ),
        "utterance_id": [item["utterance_id"] for item in dataset_items],
    }

    return result_batch
