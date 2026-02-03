param(
  [Parameter(Mandatory=$false)][string]$Ckpt = "outputs/val_sft_stageB_best11500_mask_s800/r3m_rec_final.pt",
  [Parameter(Mandatory=$false)][int]$KSteps = 2,
  [Parameter(Mandatory=$false)][int]$MaxNew = 160,
  [Parameter(Mandatory=$false)][double]$Temperature = 0.8,
  [Parameter(Mandatory=$false)][int]$TopK = 50,
  [Parameter(Mandatory=$false)][double]$RepetitionPenalty = 1.1,
  [Parameter(Mandatory=$false)][string]$System = "You are a helpful assistant."
)

$ErrorActionPreference = "Stop"

# Runs the chat REPL inside your WSL repo checkout (where checkpoints live).
# Usage (from Windows PowerShell in the repo `scripts/` dir):
#   .\chat_r3m_ckpt_wsl.ps1 -Ckpt outputs/val_sft_stageB_best11500_mask_s800/r3m_rec_final.pt -KSteps 2

& wsl -e bash -lc "cd ~/BBotS1-wsl && source .venv/bin/activate && python -u scripts/chat_r3m_ckpt.py --ckpt $Ckpt --k-steps $KSteps --max-new $MaxNew --temperature $Temperature --top-k $TopK --repetition-penalty $RepetitionPenalty --system `"$System`""

