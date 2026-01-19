from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import DatasetDict

class AudioDataLoader:
    def __init__(self, data_root, model_name="facebook/wav2vec2-base", max_duration=5.0):
        self.data_root = data_root
        self.max_duration = max_duration
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def _load_dataset(self):
        return load_dataset("audiofolder", data_dir=self.data_root)

    def _preprocess(self, examples):
        audio_arrays = [x["array"] for x in examples["audio"]]

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        inputs["labels"] = examples["label"]
        return inputs

    def get_dataloader(self,split = "train", batch_size=8):
        dataset = self._load_dataset()

        if split == "test":
            test_only_dataset = DatasetDict({
                "test": dataset["test"]
            })

            dataset = test_only_dataset.map(
                self._preprocess,
                batched=True,
                remove_columns=["audio", "label"]
            )

            test_loader = DataLoader(
                dataset["test"],
                batch_size=batch_size,
                collate_fn=default_data_collator
            )
        
            return test_loader
        
        if split == "train":
            train_test_dataset = DatasetDict({
                "train": dataset["train"],
            })
            dataset = train_test_dataset.map(
                self._preprocess,
                batched=True,
                remove_columns=["audio", "label"]
            )

            train_loader = DataLoader(
                dataset['train'],
                batch_size=batch_size,
                collate_fn=default_data_collator
            )
        
            return train_loader

        if split == "validation":
            validation_only_dataset = DatasetDict({
                "validation": dataset["validation"]
            })

            dataset = validation_only_dataset.map(
                self._preprocess,
                batched=True,
                remove_columns=["audio", "label"]
            )

            validation_loader = DataLoader(
                dataset["validation"],
                batch_size=batch_size,
                collate_fn=default_data_collator
            )
        
            return validation_loader