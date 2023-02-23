import os
import json, yaml
from PIL import Image
from autocrop import Cropper

def crop_face(input_dir, output_dir, extensions=['.jpg', '.jpeg', '.png']):
    """
    Crop faces in images within the input directory and save the results in the output 
    directory.

    Args:
        input_dir (str): Path to the input directory containing images to crop.
        output_dir (str): Path to the output directory to save cropped images.

    Returns:
        tuple: A tuple containing the number of images that were cropped and the list of 
                filenames that failed to crop.
    """
    os.makedirs(output_dir, exist_ok=True)

    cropper = Cropper()
    cropped, failed = 0, []

    for filename in os.listdir(input_dir):
        if os.path.splitext(filename)[1] in extensions:
            try:
                image = Image.fromarray(cropper.crop(os.path.join(input_dir, filename)))
                image.save(os.path.join(output_dir, filename))
                cropped += 1
            except:
                failed.append(filename)

    print(f"Crop image: {cropped} / {cropped + len(failed)} in {input_dir}", f", failed: {failed}" if failed else "")

    return cropped, failed


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f).get("dreambooth")

    # setup folder structures, crop face, and create concepts 
    root_dir = f"/dreambooth/data"
    input_dir = f"{root_dir}/input"
    output_dir = f"{root_dir}/output"
    concepts = []

    for instance in config.get("instances"):
        image_dir = f"{input_dir}/{instance.get('id')}"
        cropped_image_dir = f"{image_dir}/cropped"

        os.makedirs(image_dir, exist_ok=True)   # only 2 deepest folders need to be created
        os.makedirs(output_dir, exist_ok=True)
        os.system("chmod -R 777 /dreambooth/data")

        concepts.append({
            "instance_prompt":      instance.get("instance_prompt"),
            "class_prompt":         instance.get("class_prompt"),
            "instance_data_dir":    cropped_image_dir if instance.get("crop_images") else image_dir,
            "class_data_dir":       f"{output_dir}/class-data/{instance.get('id')}"
        })

    with open(f"{input_dir}/concepts.json", "w") as f:
        json.dump(concepts, f, indent=4)

    input("Configurations completed, please put corresponded images in following folders"
          f'{[c.get("instance_data_dir") for c in concepts]}\n'
          "Enter/Return key to continue, or ctrl+c to abort: ")

    # crop face
    for instance in config.get("instances"):
        image_dir = f"{input_dir}/{instance.get('id')}"
        cropped_image_dir = f"{image_dir}/cropped"

        if instance.get("crop_face"):
            cropped, _ = crop_face(image_dir, cropped_image_dir)
            
            if cropped == 0:
                raise Exception(f"No image was cropped in {image_dir}, please check the images and try again.")

    # construct commands
    def append_optional_params(cmd, param, flag_only=False):
        if config.get(param) != None:
            cmd.append(
                f"--{param}" if flag_only else \
                f"--{param}=\"{config.get(param)}\"")

    train_command = [ "accelerate launch train_dreambooth.py",
        f'--pretrained_model_name_or_path="{config.get("pretrained_model_name_or_path")}"', 
        f'--concepts_list="{input_dir}/concepts.json"', 
        f'--output_dir="{output_dir}"', 
        f'--max_train_steps={config.get("max_train_steps")}', 
        f'--save_sample_prompt="{config.get("save_sample_prompt")}"', 
        f'--save_interval={config.get("save_interval")}', 
        f'--sample_batch_size={config.get("sample_batch_size")}',
        f'--resolution={config.get("resolution")}', 
        f'--train_batch_size={config.get("train_batch_size")}', 
        f'--gradient_accumulation_steps={config.get("gradient_accumulation_steps")}', 
        f'--learning_rate={config.get("learning_rate")}', 
        f'--lr_scheduler={config.get("lr_scheduler")}', 
        f'--lr_warmup_steps={config.get("lr_warmup_steps")}', 
        f'--seed={config.get("seed")}',
    ]

    append_optional_params(train_command, "pretrained_vae_name_or_path")
    append_optional_params(train_command, "with_prior_preservation", True)
    append_optional_params(train_command, "prior_loss_weight")
    # append_optional_params(train_command, "seed")
    append_optional_params(train_command, "train_text_encoder", True)
    append_optional_params(train_command, "mixed_precision")
    append_optional_params(train_command, "use_8bit_adam", True)
    append_optional_params(train_command, "num_class_images")

    train_command = " \\\n    ".join(train_command)

    ckpt_filename = '{"-".join([i.get("id") for i in config.get("instances")])}-{config.get("max_train_steps")}.ckpt'
    ckpt_filepath = f'{output_dir}/{ckpt_filename}'
    convert_ckpt_command = [ "python3 convert_diffusers_to_original_stable_diffusion.py",
        f'--model_path="{output_dir}/{config.get("max_train_steps")}"',
        f'--checkpoint_path="{ckpt_filepath}"', 
        "--half"
    ]
    convert_ckpt_command = " \\\n    ".join(convert_ckpt_command)
    move_ckpts_command = f'mv {ckpt_filepath} {config.get("sd_model_path")}/{ckpt_filename}'

    if config.get("dry_run"):
        print("\nTraining command \n================\n" + train_command)
        print("\nConvert to CKPT command \n================\n" + convert_ckpt_command)

        if config.get("sd_model_path"):
            print("\nMove CKPT command \n================\n" + move_ckpts_command)
    else:
        status = os.system(train_command)

        if status == 0:
            status = os.system(convert_ckpt_command)
        else:
            print("\nTraining failed, command \n================\n" + train_command)

        if status == 0:
            if config.get("sd_model_path"):
                status = os.system(move_ckpts_command)
            else:
                print(f"\nTraining complete, and convert to: {ckpt_filepath}")
        else:
            print("\nConvert to CKPT failed, command \n================\n" + convert_ckpt_command)

        if status == 0:
            print(f'\n{ckpt_filepath} moved to {config.get("sd_model_path")}')
        else:
            print("\nMoving ckpt file failed, command \n================\n" + move_ckpts_command)
        
