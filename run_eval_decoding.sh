CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=boolq --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=piqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=siqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=hellas --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=winog --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=arce --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=arcc --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4
CUDA_VISIBLE_DEVICES=0 python train_hira.py --dataset=obqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4

# if you have multiple gpus, you can:
#CUDA_VISIBLE_DEVICES=0 python mul_lora_hg.py --dataset=boolq --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/boolq.log 2>&1 &&
#CUDA_VISIBLE_DEVICES=0 python mul_lora_hg.py --dataset=piqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/piqa.log 2>&1 &&
#CUDA_VISIBLE_DEVICES=0 python mul_lora_hg.py --dataset=siqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/siqa.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=1 python mul_lora_hg.py --dataset=hellas --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/hellas.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=2 python mul_lora_hg.py --dataset=winog --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/winog.log 2>&1 &&
#CUDA_VISIBLE_DEVICES=2 python mul_lora_hg.py --dataset=arce --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/arce.log 2>&1 &
#
#CUDA_VISIBLE_DEVICES=3 python mul_lora_hg.py --dataset=arcc --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/arcc.log 2>&1 &&
#CUDA_VISIBLE_DEVICES=3 python mul_lora_hg.py --dataset=obqa --batch=16 --output_folder=temp --ckpt=$1 --beam_size=4 > logs/obqa.log 2>&1 &
