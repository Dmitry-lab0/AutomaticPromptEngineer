from model_inference import LLMInference


def calc_metrics(confusion_matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """
    Рассчитывает метрики качества (precision, recall, F1, accuracy) на основе матрицы ошибок.

    Args:
        confusion_matrix: Словарь с ключами-полями и значениями в виде словарей с ключами:
                          'tp', 'tn', 'fp', 'fn', 'fpn'.

    Returns:
        Словарь с метриками для каждого поля и усреднённые макро-метрики.
    """
    accuracy_precision_recall = {}
    for field, counts in confusion_matrix.items():
        tp = counts["tp"]
        tn = counts["tn"]
        fp = counts["fp"]
        fn = counts["fn"]
        fpn = counts["fpn"]

        total = tp + tn + fn + fp + fpn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp + fpn) if (tp + fp + fpn) > 0 else 0
        recall = tp / (tp + fn + fpn) if (tp + fn + fpn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        accuracy_precision_recall[field] = (accuracy, precision, recall, f1)

    metrics_dict = {}
    f1_macro = 0.0
    precision_macro = 0.0
    recall_macro = 0.0

    for field_name, (accuracy, precision, recall, f1) in accuracy_precision_recall.items():
        metrics_dict[field_name] = {
            "f1": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
        f1_macro += f1
        precision_macro += precision
        recall_macro += recall

    num_fields = len(accuracy_precision_recall)
    metrics_dict["macro avg"] = {
        "f1": round(f1_macro / num_fields, 3),
        "precision": round(precision_macro / num_fields, 3),
        "recall": round(recall_macro / num_fields, 3),
    }

    return metrics_dict

def calc_confusion_matrix_with_llm_batch(
    model_inference: LLMInference,  
    prompt_template,
    answers: dict[str, dict[str, str]],
    processed_gt_markup: dict[str, dict[str, str]],
    barcodes: list[str],
    fields: dict[str, str],
) -> dict[str, dict[str, int]]:
    """
    Рассчитывает матрицу ошибок с использованием LLM для сложных случаев.

    Args:
        model_inference: Объект модели, поддерживающий методы `run` и `parse_llm_comparison_output`.
        prompt_template: Функция, формирующая промпт для LLM.
        answers: Предсказания модели в формате {barcode: {field: value}}.
        processed_gt_markup: Ground truth в формате {barcode: {field: value}}.
        barcodes: Список баркодов документов.
        fields: Словарь сущностей в формате {field_name: description}.

    Returns:
        Матрица в формате:
        {
            field: {
                'tp': int, 'tn': int, 'fp': int, 'fn': int, 'fpn': int
            },
            ...
        }
    """
    confusion_matrix = {
        key: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "fpn": 0} for key in fields
    }

    prompts = []
    prompt_meta = []

    for barcode in barcodes:
        markup_doc = processed_gt_markup[barcode]
        prediction_doc = answers[barcode]

        for field, description in fields.items():
            prediction = prediction_doc[field].lower()
            markup = markup_doc[field].lower()

            if markup == prediction:
                confusion_matrix[field]["tp" if prediction else "tn"] += 1
            else:
                if prediction:
                    if markup:
                        prompt = prompt_template(field, description, prediction, markup)
                        prompts.append(prompt)
                        prompt_meta.append((barcode, field))
                    else:
                        confusion_matrix[field]["fp"] += 1
                else:
                    confusion_matrix[field]["fn"] += 1

    outputs = model_inference.run(prompts)

    for (barcode, field), output_text in zip(prompt_meta, outputs):
        judgment = model_inference.parse_llm_comparison_output(output_text)
        if judgment == "false":
            confusion_matrix[field]["fpn"] += 1
        else:
            confusion_matrix[field]["tp"] += 1

    return confusion_matrix
