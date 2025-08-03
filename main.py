import argparse
from typing import Any

from ape import AutomaticPromptEngineer
from create_prompts import get_prompts_templates
from dataset import DatasetManager
from model_inference import load_models_from_cfg
from utils import get_config, load_json, create_folder


def main(cfg_path: str, prompts_cfg_path: str, exp_name: str) -> Any:
    # LOAD CONFIGS
    cfg = get_config(cfg_path)
    prompts_cfg = get_config(prompts_cfg_path)
    cfg = cfg["experiments"][exp_name]

    # LOAD INPUT
    dataset = load_json(cfg["dataset_path"])
    dataset = DatasetManager(dataset, cfg["random_seed"])

    target_prompt_template, task_description_init, task_description_iterative, calc_metrics_template = get_prompts_templates(
        prompts_cfg["target_prompt_template"],
        prompts_cfg["task_description_init"],
        prompts_cfg["task_description_iterative"],
        prompts_cfg["calc_metrics_prompt"],
    )

    model_instances = load_models_from_cfg(cfg)
    ape = AutomaticPromptEngineer(
        cfg["doc_names"],
        dataset,
        target_prompt_template,
        task_description_init,
        task_description_iterative,
        calc_metrics_template,
        model_instances["prompt_gen_model"],
        model_instances["metric_calc_model"],
        model_instances["eval_model"],
    )

    create_folder(cfg["results_folder_name"])
    optimized_prompt = ape.optimize_prompt(
        cfg["n_iterations"],
        cfg["n_train_sample"],
        cfg["resample_each_n"],
        cfg["results_folder_name"],
    )

    return optimized_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Запуск основного скрипта с параметрами"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Путь к конфигурационному файлу основного конфига",
    )
    parser.add_argument(
        "--prompts_cfg_path",
        type=str,
        required=True,
        help="Путь к конфигурационному файлу для промптов",
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Имя эксперимента")

    args = parser.parse_args()

    optimized_prompt = main(
        cfg_path=args.cfg_path,
        prompts_cfg_path=args.prompts_cfg_path,
        exp_name=args.exp_name,
    )
    print(optimized_prompt)
