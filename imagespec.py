import os
from PIL import Image

def resize_images(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    count = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize((target_width, target_height))
                    resized_img.save(output_path)
                    count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"✅ {count} image(s) resized and saved to '{output_folder}'.")


if __name__ == "__main__":
    input_folder = input("Enter path to input folder: ").strip()
    output_folder = input("Enter path to output folder: ").strip()
    width = int(input("Enter new width: ").strip())
    height = int(input("Enter new height: ").strip())

    if not os.path.exists(input_folder):
        print("❌ Input folder does not exist.")
    else:
        resize_images(input_folder, output_folder, width, height)
