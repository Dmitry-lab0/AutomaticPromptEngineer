import yaml
import json
import os

from typing import Any, Optional


def load_json(path: str) -> Any:
    """
    Загружает JSON-файл по указанному пути.

    Args:
        path (str): Путь к JSON-файлу.

    Returns:
        dict[str, Any]: Содержимое JSON-файла в виде словаря.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config(cfg_path: str) -> Any:
    """
    Загружает конфигурационный файл в формате YAML.

    Args:
        cfg_path (str): Путь к конфигурационному файлу.

    Returns:
        dict[str, Any]: Конфигурация в виде словаря.
    """
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(save_path: str, outputs: dict[str, Any]) -> None:
    """
    Сохраняет словарь в JSON-файл по указанному пути.

    Args:
        save_path (str): Путь для сохранения JSON-файла.
        outputs (dict[str, Any]): Данные для сохранения.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)


def create_folder(path: str) -> None:
    """
    Создаёт директорию по указанному пути, если она не существует.

    Args:
        path (str): Путь к директории.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Папка '{path}' успешно создана.")
    except Exception as e:
        print(f"Ошибка при создании папки '{path}': {e}")


def get_proxy_access_token(secret_name: str = "ACCESS_TOKEN") -> Optional[str]:
    """
    Получает значение токена из переменных окружения.

    Args:
        secret_name (str): Название переменной окружения.

    Returns:
        Optional[str]: Значение переменной окружения или None, если не найдено.
    """
    return os.environ.get(secret_name)
