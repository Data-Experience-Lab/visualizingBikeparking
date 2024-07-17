import os
import cv2
from PIL import Image

input_dir = "D:\\RLHS-MITACS\\Data\\cleaned_data"
output_dir  = "D:\\RLHS-MITACS\\Data\\cleaned_data_rn"\

images = os.listdir(input_dir)
images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))
print(images)

# x1,y1,x2,y2 = ()

# for image_file in images:
#     image = cv2.imread(os.path.join(dir_path, image_file))
#     new_image = image[x1:x2, y1:y2]
#     cv2.imwrite(os.path.join(new_dir, image_file))

def center_crop_and_resize(image_path, output_path, size=(640, 640)):
    with Image.open(image_path) as img:
        width, height = img.size
        new_size = min(width, height)
        

        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        
        
        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(output_path)



# def crop(main_directory, output_directory):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
        
#     image_counter = 1 #to start image names with 1.png,2.png...

#     for subdir, _, files in os.walk(main_directory):
#         for filename in files:
#             if filename.lower().endswith(('png', 'jpg', 'jpeg')):
#                 input_path = os.path.join(subdir, filename)
#                 output_path = os.path.join(output_directory, f"{image_counter}.png")
#                 center_crop_and_resize(input_path, output_path)
#                 image_counter += 1

def crop(input_dir, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    image_counter = 1 #to start image names with 1.png,2.png...

    for image in images:       
        input_path = os.path.join(input_dir, image)
        output_path = os.path.join(output_directory, f"{image_counter}.png")
        center_crop_and_resize(input_path, output_path)
        image_counter += 1


                

crop(input_dir,output_dir)