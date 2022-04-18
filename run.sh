## record demos
python record_demos.py --seed=42 --device="cpu" \
 --env="HalfCheetah-v3" \
 --model_path="pretrained/sac_forward" \
 --num_episodes=211 \
 --save_path="data"

python record_demos.py --seed=42 --device="cpu" \
 --env="BackflipCheetah-v0" \
 --model_path="pretrained/sac_backflip" \
 --num_episodes=211 \
 --save_path="data"

# convert to robomimic format
python convert_to_robomimic.py \
  --seed=42 \
  --env="HalfCheetah-v3" \
  --data_path="data/HalfCheetah-v3.pkl" \
  --save_path="data" \
  --num_valid=10

python convert_to_robomimic.py \
  --seed=42 \
  --env="BackflipCheetah-v0" \
  --data_path="data/BackflipCheetah-v0.pkl" \
  --save_path="data" \
  --num_valid=10
