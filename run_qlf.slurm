#!/bin/bash
#SBATCH --job-name=QLearningFourier
#SBATCH --ntasks=3
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=compute-fit
#SBATCH --output=%u_%x_%j.out

cd TesisFinal

# Activar módulo de Python 3.12
module load softwares Python/3.12.3-GCCcore-13.3.0

# Ejecutar el script
python3.12 RunQLF.py