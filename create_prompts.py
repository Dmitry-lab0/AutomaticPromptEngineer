from typing import Any, Callable, Optional


def dict_discriptions_to_list(descriptions: dict[str, str]) -> list[str]:
    """
    Преобразует словарь описаний в список строк формата "ключ|значение".

    Args:
        descriptions: Словарь, где ключ — имя поля, значение — описание.

    Returns:
        Список строк в формате "ключ|значение".
    """
    return [f"{field_name}|{disc}" for field_name, disc in descriptions.items()]


def join_entities(entities: list[str]) -> str:
    """
    Преобразует список сущностей в строку с перечислением через минус.

    Args:
        entities: Список строк, представляющих сущности.

    Returns:
        Строка в формате "- сущность1\n- сущность2".
    """
    return "\n".join(["- " + entity for entity in entities])


def add_examples(
    inputs: list[str],
    outputs: list[Any],
    preds: Optional[list[Any]] = None,
) -> str:
    """
    Формирует строку с примерами входных данных, ожидаемых и предсказанных результатов.

    Args:
        inputs: Список входных текстов.
        outputs: Список ожидаемых результатов (ground truth).
        preds: Опционально, список предсказанных результатов.

    Returns:
        Строка с объединёнными примерами.
    """
    lines = []
    for idx, (input_text, output) in enumerate(zip(inputs, outputs)):
        block = (
            f"Текст документа {idx+1}:\n{input_text}\n"
            f"Разметка {idx+1}:\n{output}"
        )
        if preds:
            block += f"\nПредсказанный ответ для документа {idx+1}:\n{preds[idx]}"
        lines.append(block)
    return "\n".join(lines)


def join_desriptions_with_scores(
    descriptions: dict[str, list[str]],
    scores: dict[str, float],
) -> str:
    """
    Формирует строку с описаниями и соответствующими F1-метриками.

    Args:
        descriptions: Словарь описаний, где ключ — имя поля, значение — список описаний.
        scores: Словарь с метриками, где ключ — имя поля, значение — F1-оценка.

    Returns:
        Строка с объединёнными описаниями и метриками.
    """
    return "\n".join([f"{field_name}|{disc}. F1 поля: {scores[field_name]}" for field_name, disc in descriptions.items()])


def get_target_prompt_template(prompt: str) -> Callable[[str, dict[str, str]], str]:
    """
    Возвращает функцию для формирования целевого промпта.

    Args:
        prompt: Шаблон промпта с плейсхолдерами {query} и {entities}.

    Returns:
        Функция, принимающая query и entities и возвращающая готовый промпт.
    """
    def wrapper(query: str, entities: dict[str, str]) -> str:
        return prompt.format(
            query=query.strip(),
            entities=join_entities(dict_discriptions_to_list(entities)),
        )

    return wrapper

def get_task_description_init_template(prompt: str) -> Callable[[list[str], list[Any]], str]:
    """
    Возвращает функцию для формирования начального описания задачи.

    Args:
        prompt: Шаблон промпта с плейсхолдерами {entities_names} и {examples}.

    Returns:
        Функция, принимающая entities_names и inputs (и опционально outputs), возвращающая строку.
    """
    def wrapper(entities_names: list[str], inputs: list[Any], outputs: list[Any] | None = None) -> str:
        return prompt.format(
            entities_names=join_entities(entities_names),
            examples=add_examples(inputs, outputs),
        )

    return wrapper

def get_task_description_iter_template(prompt: str) -> Callable[
    [dict[str, str], dict[str, float], list[str], list[Any], Optional[list[Any]]],
    str,
]:
    """
    Возвращает функцию для формирования итеративного описания задачи.

    Args:
        prompt: Шаблон промпта с плейсхолдерами {entities_names_discriptions} и {examples_with_preds}.

    Returns:
        Функция, принимающая описания, метрики, входы, выходы и предсказания, возвращающая строку.
    """
    def wrapper(
        entities_names_discriptions: dict[str, str],
        scores: dict[str, float],
        inputs: list[str],
        outputs: list[Any],
        preds: Optional[list[Any]] = None,
    ) -> str:
        return prompt.format(
            entities_names_discriptions=join_desriptions_with_scores(
                entities_names_discriptions, scores
            ),
            examples_with_preds=add_examples(inputs, outputs, preds),
        )

    return wrapper


def get_prompts_templates(
    target_prompt_str: str,
    task_description_init_str: str,
    task_description_iterative_str: str,
    calc_metrics_template_str: str,
) -> tuple[
    Callable[[str, dict[str, str]], str],
    Callable[[list[str], list[Any]], str],
    Callable[[dict[str, str], dict[str, float], list[str], list[Any], Optional[list[Any]]], str],
    Callable[[str, str, str, str], str],
]:
    """
    Возвращает набор функций-шаблонов для генерации промптов.

    Args:
        target_prompt_str: Шаблон целевого промпта.
        task_description_init_str: Шаблон начального описания задачи.
        task_description_iterative_str: Шаблон итеративного описания задачи.
        calc_metrics_template_str: Шаблон для расчёта метрик.

    Returns:
        Кортеж из четырёх функций-шаблонов.
    """
    target_prompt_template = get_target_prompt_template(target_prompt_str)
    task_description_init = get_task_description_init_template(
        task_description_init_str
    )
    task_description_iterative = get_task_description_iter_template(
        task_description_iterative_str
    )
    calc_metrics_template = get_llm_metrics_calc_template(calc_metrics_template_str)
    return (
        target_prompt_template,
        task_description_init,
        task_description_iterative,
        calc_metrics_template,
    )


def get_llm_metrics_calc_template(prompt: str) -> Callable[[str, str, str, str], str]:
    """
    Возвращает функцию для формирования промпта оценки метрик.

    Args:
        prompt: Шаблон промпта с плейсхолдерами {desc}, {prediction}, {ground_truth}.

    Returns:
        Функция, принимающая описание, предсказание и ground truth, возвращающая строку.
    """
    return lambda field, description, prediction, ground_truth: prompt.format(
        desc=field + "|" + description, prediction=prediction, ground_truth=ground_truth
    )
