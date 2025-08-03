import random
from typing import Any


class DatasetManager:
    REQUIRED_KEYS = {"barcode", "text", "gt"}

    def __init__(self, dataset: dict[str, list[dict]], seed: int):
        """
        Инициализация менеджера датасета.

        Args:
            dataset: Словарь с ключами 'train', 'val', 'test', каждый из которых — список словарей.
            seed: Сид для воспроизводимости.
        """
        random.seed(seed)

        self._validate_split(dataset.get("train"), "train")
        self._validate_split(dataset.get("val"), "val")
        self._validate_split(dataset.get("test"), "test")

        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["val"]
        self.test_dataset = dataset["test"]

    def _validate_split(self, split: Any, split_name: str) -> None:
        """
        Проверяет, что раздел датасета соответствует ожидаемой структуре.

        Args:
            split: Данные (например, train/val/test).
            split_name: Название для вывода ошибок.

        Raises:
            ValueError: Если структура раздела некорректна.
        """
        if not isinstance(split, list):
            raise ValueError(f"'{split_name}' dataset must be a list of dicts.")
        for i, entry in enumerate(split):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {i} in '{split_name}' must be a dict.")
            missing = self.REQUIRED_KEYS - entry.keys()
            if missing:
                raise ValueError(
                    f"Entry {i} in '{split_name}' is missing keys: {missing}"
                )

    def get_texts(self, dataset_split: list[dict]) -> list[str]:
        """
        Возвращает список текстов из раздела датасета.

        Args:
            dataset_split: Список словарей, представляющих документы.

        Returns:
            Список текстов.
        """
        return [doc["text"] for doc in dataset_split]

    def get_barcodes(self, dataset_split: list[dict]) -> list[str]:
        """
        Возвращает список баркодов из раздела датасета.

        Args:
            dataset_split: Список словарей, представляющих документы.

        Returns:
            Список баркодов.
        """
        return [doc["barcode"] for doc in dataset_split]

    def get_gt(self, dataset_split: list[dict]) -> dict[str, Any]:
        """
        Возвращает ground truth в виде словаря {barcode: gt}.

        Args:
            dataset_split: Список словарей, представляющих документы.

        Returns:
            Словарь с ground truth.
        """
        return {doc["barcode"]: doc["gt"] for doc in dataset_split}

    def get_train(self) -> list[dict]:
        """
        Возвращает обучающий набор данных.

        Returns:
            Список словарей с данными.
        """
        return self.train_dataset

    def get_test(self) -> list[dict]:
        """
        Возвращает тестовый набор данных.

        Returns:
            Список словарей с данными.
        """
        return self.test_dataset

    def get_val(self) -> list[dict]:
        """
        Возвращает валидационный набор данных.

        Returns:
            Список словарей с данными.
        """
        return self.val_dataset

    def get_shuffled_train_chunks(self, n_train_sample: int) -> list[list[dict]]:
        """
        Возвращает перемешанные чанки обучающего набора.

        Args:
            n_train_sample: Размер каждого чанка.

        Returns:
            Список списков (чанков) документов.
        """
        shuffled = self.train_dataset.copy()
        random.shuffle(shuffled)
        chunks = [
            shuffled[i:i + n_train_sample]
            for i in range(0, len(shuffled), n_train_sample)
        ]
        return chunks

    def get_chunk_barcodes(self, chunks: list[list[dict]]) -> list[list[str]]:
        """
        Возвращает баркоды для каждого чанка.

        Args:
            chunks: Список чанков (списков документов).

        Returns:
            Список списков баркодов.
        """
        return [[doc["barcode"] for doc in chunk] for chunk in chunks]
