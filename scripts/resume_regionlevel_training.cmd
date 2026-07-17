@echo off
setlocal

set "ROOT_DIR=D:\ChenMeng\Graduate_Student\Experiment_and_Data\Flare_with_without_CME_forecast"
set "WANDB_DIR=%ROOT_DIR%\wandb"
set "WANDB_CACHE_DIR=%ROOT_DIR%\wandb\cache"
set "WANDB_CONFIG_DIR=%ROOT_DIR%\wandb\config"
set "WANDB_DATA_DIR=%ROOT_DIR%\wandb\data"
set "PYTHONIOENCODING=utf-8"

cd /d "%ROOT_DIR%"

"C:\TOOLs\Anoconda3_2024.06-1\envs\Flare_with_without_CME_forecast\python.exe" ^
  scripts\f_train_model.py ^
  --data_config configs/data_config_stage2_yolo11_mag_euv171_euv94_256.yaml ^
  --model_config configs/model_config.yaml ^
  --train_config configs/training_config_stage2_yolo11_regionlevel.yaml ^
  --output_dir outputs/stage2_yolo11_regionlevel ^
  --log_dir logs ^
  --resume_latest ^
  >> "%ROOT_DIR%\logs\resume_regionlevel_console.log" 2>&1

endlocal
