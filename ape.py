from collections import defaultdict
from functools import partial
from typing import Any, Callable, DefaultDict, Optional

from dataset import DatasetManager
from model_inference import LLMInference

from calc_metrics import calc_confusion_matrix_with_llm_batch, calc_metrics
from utils import save_json


class AutomaticPromptEngineer:
    def __init__(
        self,
        entities_names: list[str],
        dataset: DatasetManager,
        target_prompt_template: Callable[[str, dict[str, str]], str],
        task_description_init: Callable[[list[str], list[Any]], str],
        task_description_iterative: Callable[[dict[str, str], dict[str, float], list[str], list[Any], Optional[list[Any]]], str],
        calc_metrics_template: Callable[[str, str, str, str], str],
        prompt_gen_llm: LLMInference,
        metric_calc_llm: LLMInference,
        eval_llm: LLMInference,
    ):
        """
        Инициализация движка автоматической генерации промптов.

        Args:
            entities_names: Список имён сущностей, которые нужно извлечь.
            dataset: Датасет.
            target_prompt_template: Функция, формирующая финальный промпт.
            task_description_init: Функция для генерации описания задачи на начальном этапе.
            task_description_iterative: Функция для генерации описания задачи на следующих этапах.
            calc_metrics_template: Функция для формирования промпта для сравнения разметки и предсказания.
            prompt_gen_llm: Модель для генерации промптов.
            metric_calc_llm: Модель для сравнения разметки и предсказания.
            eval_llm: Модель для NER.
        """
        self.entities_names = entities_names
        self.target_prompt_template = target_prompt_template
        self.task_description_init = task_description_init
        self.task_description_iterative = task_description_iterative
        self.calc_metrics_template = calc_metrics_template
        self.dataset = dataset

        # LLMs
        self.prompt_gen_llm = prompt_gen_llm
        self.metric_calc_llm = metric_calc_llm
        self.eval_llm = eval_llm

        self.best_scores: DefaultDict[str, float] = defaultdict(lambda: -float("inf"))
        self.best_descriptions: DefaultDict[str, str]  = defaultdict(lambda: "")
        self.train_preds: DefaultDict[str, dict] = defaultdict(lambda: {})

    def generate_candidate_prompts(
        self, task_descriptions: list[str]
    ) -> list[dict[str, str]]:
        """
        Генерирует кандидатов-промптов на основе описаний задач.

        Args:
            task_descriptions: Список описаний задач для генерации промптов.

        Returns:
            Список сгенерированных кандидатов-промптов.
        """
        print("Generating prompts...")
        outputs = self.prompt_gen_llm.run(task_descriptions)
        candidates = []
        for text in outputs:
            parsed = self.prompt_gen_llm.parse_llm_response(text, self.entities_names)
            candidates.append(parsed)

        print("[INFO] Prompt generation done.")
        return candidates

    def create_user_msgs(
        self, texts: list[str], prompt_template: Callable[[str], str]
    ) -> list[str]:
        """
        Создаёт список пользовательских сообщений на основе текстов и шаблона.

        Args:
            texts: Список входных текстов.
            prompt_template: Шаблон промпта.

        Returns:
            Список сформированных пользовательских сообщений.
        """
        return [prompt_template(text) for text in texts]

    def evaluate_prompts(
        self, candidate: dict[str, str], dataset: str
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        """
        Оценивает качество промпта на указанном наборе данных.

        Args:
            candidate: Кандидат-промпт в виде словаря {entity: description}.
            dataset: Название набора данных.

        Returns:
            Кортеж из метрик и предсказаний модели.
        """
        fixed_prompt = partial(self.target_prompt_template, entities=candidate)
        texts = self.dataset.get_texts(dataset)
        user_prompts = self.create_user_msgs(texts, fixed_prompt)
        barcodes = self.dataset.get_barcodes(dataset)

        ner_results = self.eval_llm.run(user_prompts)

        parsed_data = {
            barcode: self.eval_llm.parse_llm_response(output_llm, self.entities_names)
            for output_llm, barcode in zip(ner_results, barcodes)
        }
        gt = self.dataset.get_gt(dataset)
        confusion_matrix = calc_confusion_matrix_with_llm_batch(
            self.metric_calc_llm,
            self.calc_metrics_template,
            parsed_data,
            gt,
            barcodes,
            candidate,
        )
        metrics = calc_metrics(confusion_matrix)
        print("[INFO] Prompt evaluation done.")
        return metrics, parsed_data

    def get_task_descriptions(self, train_chunks: list[list[dict]]) -> list[str]:
        """
        Генерирует описания задач для каждого чанка обучающих данных.

        Args:
            train_chunks: Список чанков обучающих данных.

        Returns:
            Список описаний задач.
        """
        task_descriptions = []
        for chunk_idx, train_samples in enumerate(train_chunks):
            train_texts = self.dataset.get_texts(train_samples)
            train_outputs = list(self.dataset.get_gt(train_samples).values())
            train_chunks_barcodes = self.dataset.get_chunk_barcodes(train_chunks)
            if self.best_descriptions:
                train_preds_chunk = [
                    self.train_preds[barcode]
                    for barcode in train_chunks_barcodes[chunk_idx]
                ]
                task_description = self.task_description_iterative(
                    self.best_descriptions,
                    self.best_scores,
                    train_texts,
                    train_outputs,
                    train_preds_chunk,
                )
            else:
                task_description = self.task_description_init(
                    self.entities_names, train_texts, train_outputs
                )
            task_descriptions.append(task_description)
        return task_descriptions

    def upd_best_scores_and_best_descriptions(
        self, scores: dict[str, dict[str, float]], candidate: dict[str, str]
    ) -> bool:
        """
        Обновляет лучшие оценки и описания, если найдены лучшие результаты.

        Args:
            scores: Словарь с текущими метриками по сущностям.
            candidate: Кандидат-описание сущностей.

        Returns:
            True, если были обновлены лучшие оценки, иначе False.
        """
        calc_metrics_flag = False
        for entity_name in self.entities_names:
            score = scores[entity_name]["f1"]
            description_candidate = candidate[entity_name]
            best_score = self.best_scores[entity_name]

            if score > best_score:
                calc_metrics_flag = True
                self.best_scores[entity_name] = score
                self.best_descriptions[entity_name] = description_candidate

        return calc_metrics_flag

    def optimize_prompt(
        self,
        n_iterations: int = 6,
        n_train_sample: int = 2,
        resample_each_n: int = 3,
        save_folder: str = "test_without_name",
    ) -> dict[str, str]:
        """
        Основной метод генерации промптов.

        Args:
            n_iterations: Количество итераций генерации.
            n_train_sample: Количество "обучающих" примеров на чанк.
            resample_each_n: Частота перемешивания "обучающих" данных.
            save_folder: Папка для сохранения промежуточных результатов.

        Returns:
            Лучшие найденные описания для сущностей.
        """
        train_chunks = self.dataset.get_shuffled_train_chunks(n_train_sample)
        for iteration in range(n_iterations):
            print(f"Iteration {iteration}")
            if iteration % resample_each_n == 0:
                train_chunks = self.dataset.get_shuffled_train_chunks(n_train_sample)
                self.train_preds = defaultdict(lambda: "")
                print("[INFO] Resampled train dataset")

            calc_metrics_flag = False
            task_descriptions = self.get_task_descriptions(train_chunks)
            candidates = self.generate_candidate_prompts(task_descriptions)

            for candidate in candidates:
                val_dataset = self.dataset.get_val()
                scores, _ = self.evaluate_prompts(candidate, val_dataset)
                calc_metrics_flag = self.upd_best_scores_and_best_descriptions(
                    scores, candidate
                )
                print("CURRENT BEST SCORES")
                print(self.best_scores)

            if calc_metrics_flag:
                train_dataset = self.dataset.get_train()
                _, self.train_preds = self.evaluate_prompts(
                    self.best_descriptions, train_dataset
                )

            save_json(f"./{save_folder}/best_scores_{iteration}.json", self.best_scores)
            save_json(
                f"./{save_folder}/best_descriptions_{iteration}.json",
                self.best_descriptions,
            )

        return self.best_descriptions
