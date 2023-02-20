## Setup
IDENTIFIER = "ycycy"

MODEL_NAME = "SG161222/Realistic_Vision_V1.3"
instance_prompt = f"photo of {IDENTIFIER} person"
class_prompt = f"photo of a person"
sample_prompt = f"portrait of {IDENTIFIER} person in RAW photo, a close up portrait photo of brutal 45 y.o man in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
training_steps = 800

INSTANCE_DIR = f"/dreambooth/data/{IDENTIFIER}"
OUTPUT_DIR = f"{INSTANCE_DIR}/output" 
INPUT_DIR = f"{INSTANCE_DIR}/input"
concepts = f"{INPUT_DIR}/concepts.json"

# You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.
concepts_list = [
    {
        "instance_prompt":      instance_prompt,
        "class_prompt":         class_prompt,
        "instance_data_dir":    f"{INPUT_DIR}/images/cropped",
        "class_data_dir":       f"{OUTPUT_DIR}/class-data"
    },
#     {
#         "instance_prompt":      "photo of ukj person",
#         "class_prompt":         "photo of a person",
#         "instance_data_dir":    "/content/data/ukj",
#         "class_data_dir":       "/content/data/person"
#     }
]

# `class_data_dir` contains regularization images
import json, os

for concept in concepts_list:
    os.makedirs(concept["instance_data_dir"], exist_ok=True)
    os.makedirs(concept["class_data_dir"], exist_ok=True)

with open(concepts, "w") as f:
    json.dump(concepts_list, f, indent=4)

print("config complete")

# COMMAND = f'''
# accelerate launch train_dreambooth.py \
#   --pretrained_model_name_or_path="{MODEL_NAME}" \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --output_dir="{OUTPUT_DIR}" \
#   --revision="fp16" \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --seed=1337 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --mixed_precision="fp16" \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=50 \
#   --sample_batch_size=10 \
#   --max_train_steps={training_steps} \
#   --save_interval=400 \
#   --save_sample_prompt="{sample_prompt}" \
#   --concepts_list="{concepts}"
# '''

COMMAND = f'''
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="{MODEL_NAME}" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir="{OUTPUT_DIR}" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=10 \
  --max_train_steps={training_steps} \
  --save_interval=400 \
  --save_sample_prompt="{sample_prompt}" \
  --concepts_list="{concepts}"
'''


print(COMMAND)

step = training_steps
print(f"""
python3 convert_diffusers_to_original_stable_diffusion.py --model_path {OUTPUT_DIR}/{step}  --checkpoint_path {OUTPUT_DIR}/{IDENTIFIER}-{step}.ckpt --half
""")