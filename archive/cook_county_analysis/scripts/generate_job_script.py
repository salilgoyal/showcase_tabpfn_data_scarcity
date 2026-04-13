#!/usr/bin/env python3
"""
Generate SLURM job scripts from experiment config files.
"""

import argparse
import os
import sys
import yaml

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config, get_slurm_config, get_model_config


def generate_job_script(config_path, model_name, output_dir='generated'):
    """
    Generate a SLURM job script for a specific model from an experiment config.
    
    Args:
        config_path: Path to experiment config YAML file
        model_name: 'tabpfn' or 'xgboost'
        output_dir: Directory to save generated script
        
    Returns:
        Path to generated script
    """
    config = load_config(config_path)
    slurm_config = get_slurm_config(config)
    model_config = get_model_config(config, model_name)
    
    if model_config is None:
        raise ValueError(f"Model '{model_name}' is not enabled in config")
    
    experiment_name = config['experiment_name']
    paths = config.get('paths', {})
    env_config = config.get('environment', {})
    
    # Get script paths relative to scripts directory
    if model_name == 'tabpfn':
        script_name = 'run_tabpfn_experiment.py'
    elif model_name == 'xgboost':
        script_name = 'run_xgboost_experiment.py'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Build SLURM directives
    job_name = slurm_config.get('job_name', f"{model_name}-{experiment_name}")
    partition = slurm_config.get('partition', 'deho')
    time = slurm_config.get('time', '7-00:00:00')
    cpus = slurm_config.get('cpus_per_task', 8)
    mem = slurm_config.get('mem', '48G')
    gres = slurm_config.get('gres', None)
    account = slurm_config.get('account', None)
    
    # Environment setup
    python_module = env_config.get('python_module', 'python/3.12')
    cuda_module = env_config.get('cuda_module', 'cuda')
    venv_path = env_config.get('venv_path', '/scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate')
    
    # Generate script content
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output=../outfiles/{experiment_name}/{model_name}.out",
        f"#SBATCH --error=../outfiles/{experiment_name}/{model_name}.err",
        f"#SBATCH --time={time}",
        f"#SBATCH --partition={partition}",
    ]
    
    if gres:
        script_lines.append(f"#SBATCH --gres={gres}")
    
    script_lines.extend([
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ])
    
    if account:
        script_lines.append(f"#SBATCH --account={account}")
    
    script_lines.extend([
        "",
        f"# Generated from config: {os.path.basename(config_path)}",
        f"# Experiment: {experiment_name}",
        f"# Model: {model_name}",
        "",
        "echo \"==================================\"",
        f"echo \"Starting {model_name.upper()} Experiment\"",
        f"echo \"Experiment: {experiment_name}\"",
        "echo \"Start time: $(date)\"",
        "echo \"==================================\"",
        "",
        "# Activate environment",
        f"module load {python_module}",
    ])
    
    if gres and 'gpu' in gres:
        script_lines.append(f"module load {cuda_module}")
    
    script_lines.extend([
        f"source {venv_path}",
        "",
        "# Create output directories",
        f"mkdir -p ../outfiles/{experiment_name}",
        f"mkdir -p ../results/{experiment_name}",
        f"mkdir -p ../logs/{experiment_name}",
        "",
        "# Run experiment",
        f"python3 {script_name} \\",
        f"    --config ../configs/{os.path.basename(config_path)} \\",
        f"    --experiment_name {experiment_name}",
        "",
        "exit_code=$?",
        "",
        "echo \"\"",
        "echo \"==================================\"",
        f"echo \"{model_name.upper()} Experiment Complete\"",
        "echo \"End time: $(date)\"",
        "echo \"Exit code: $exit_code\"",
        "echo \"==================================\"",
        "",
        "if [ $exit_code -eq 0 ]; then",
        f"    echo \"✓ Results saved to: ../results/{experiment_name}/{model_name}.csv\"",
        "else",
        "    echo \"✗ Experiment failed!\"",
        "fi",
        "",
        "exit $exit_code",
    ])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write script
    script_filename = f"submit_{model_name}_{experiment_name}.sh"
    script_path = os.path.join(output_dir, script_filename)
    
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts from experiment configs')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config YAML file')
    parser.add_argument('--model', type=str, choices=['tabpfn', 'xgboost', 'both'], default='both',
                       help='Model to generate script for (default: both)')
    parser.add_argument('--output_dir', type=str, default='generated',
                       help='Directory to save generated scripts (default: scripts/generated)')
    
    args = parser.parse_args()
    
    # Resolve config path
    if not os.path.isabs(args.config):
        # Try relative to current directory, then relative to scripts directory
        if os.path.exists(args.config):
            config_path = os.path.abspath(args.config)
        elif os.path.exists(os.path.join('..', 'configs', args.config)):
            config_path = os.path.abspath(os.path.join('..', 'configs', args.config))
        else:
            parser.error(f"Config file not found: {args.config}")
    else:
        config_path = args.config
    
    models = ['tabpfn', 'xgboost'] if args.model == 'both' else [args.model]
    
    print(f"Generating job scripts from config: {config_path}")
    print(f"Models: {', '.join(models)}")
    print()
    
    generated_scripts = []
    for model in models:
        try:
            script_path = generate_job_script(config_path, model, args.output_dir)
            generated_scripts.append(script_path)
            print(f"✓ Generated: {script_path}")
        except Exception as e:
            print(f"✗ Failed to generate {model} script: {e}")
    
    print()
    print("="*60)
    print("Job scripts generated!")
    print("="*60)
    print("\nTo submit jobs:")
    for script_path in generated_scripts:
        print(f"  sbatch {script_path}")

