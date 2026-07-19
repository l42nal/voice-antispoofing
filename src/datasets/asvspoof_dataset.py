from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    LABEL_TO_ID = {
        "spoof": 0,
        "bonafide": 1,
    }

    def __init__(
        self,
        protocol_path,
        audio_dir,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            protocol_path (str | Path): path to an ASVspoof protocol file.
            audio_dir (str | Path): directory containing .flac audio files.
            limit (int | None): maximum number of dataset elements.
            shuffle_index (bool): whether to shuffle the index before limiting it.
            instance_transforms (dict[Callable] | None): transforms applied
                to individual dataset items.
        """
        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)

        index = self._create_index()

        super().__init__(
            index=index,
            limit=limit,
            shuffle_index=shuffle_index,
            instance_transforms=instance_transforms,
        )

    def _create_index(self):
        """
        Parse the protocol file and create the dataset index.

        Returns:
            index (list[dict]): metadata for every audio file.
        """
        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file does not exist: {self.protocol_path}"
            )

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory does not exist: {self.audio_dir}")

        index = []

        with self.protocol_path.open("r", encoding="utf-8") as protocol_file:
            for line_number, line in enumerate(protocol_file, start=1):
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) != 5:
                    raise ValueError(
                        f"Invalid protocol line {line_number}: "
                        f"expected 5 fields, got {len(fields)}. "
                        f"Line: {line}"
                    )

                (
                    speaker_id,
                    utterance_id,
                    environment_id,
                    attack_id,
                    label_name,
                ) = fields

                if label_name not in self.LABEL_TO_ID:
                    raise ValueError(
                        f"Unknown label '{label_name}' "
                        f"at protocol line {line_number}"
                    )

                audio_path = self.audio_dir / f"{utterance_id}.flac"

                if not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file from protocol does not exist: {audio_path}"
                    )

                index.append(
                    {
                        "path": str(audio_path),
                        "label": self.LABEL_TO_ID[label_name],
                        "utterance_id": utterance_id,
                        "speaker_id": speaker_id,
                        "environment_id": environment_id,
                        "attack_id": attack_id,
                    }
                )

        if not index:
            raise ValueError(
                f"No dataset items were found in protocol: {self.protocol_path}"
            )

        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        waveform, sample_rate = torchaudio.load(data_dict["path"])

        if sample_rate != 16000:
            raise ValueError(
                f"Expected sample rate 16000 Hz, got {sample_rate} Hz "
                f"for {data_dict['path']}"
            )

        if waveform.shape[0] != 1:
            raise ValueError(
                f"Expected mono audio, got {waveform.shape[0]} channels "
                f"for {data_dict['path']}"
            )

        waveform = waveform.squeeze(0)

        instance_data = {
            "audio": waveform,
            "audio_length": waveform.shape[0],
            "labels": data_dict["label"],
            "utterance_id": data_dict["utterance_id"],
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data
