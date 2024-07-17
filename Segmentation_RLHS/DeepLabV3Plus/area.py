import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def write_text_opencv(image, text, output_path):
    
    # image = cv2.imread(image_path)
    height, width, _ = image.shape
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    position = (10, 30)
    
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_path, image)
    
    
def calc_area_n(mask, out_path, prt = True):
# Load the segmentation mask
    # mask = Image.open(mask_path)
    mask = np.array(mask)

    # lower_green = np.array([0, 200, 0])
    # upper_green = np.array([50, 255, 50])

    # green_mask = cv2.inRange(mask, lower_green, upper_green)
    # mask[green_mask > 0] = [0, 255, 0]
    # colormap[8] = [107, 142, 35]
    # colormap[9] = [152, 251, 152]
    
    binary_mask_vegetation = np.all(mask == [107, 142, 35], axis=-1).astype(np.uint8)
    binary_mask_terrain = np.all(mask == [152, 251, 152], axis=-1).astype(np.uint8)
    # plt.imshow(binary_mask)
    # cv2.imwrite('mask.png', binary_mask)

    contours_veg, _ = cv2.findContours(binary_mask_vegetation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_terrain, _ = cv2.findContours(binary_mask_terrain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_image_veg = mask.copy()
    contour_image_ter = mask.copy()
    
    
    cv2.drawContours(contour_image_veg, contours_veg, -1, (80, 80, 80), 2)
    cv2.drawContours(contour_image_ter, contours_terrain, -1, (80, 80, 80), 2)
    
    # write_text_opencv(contour_image_veg, out_path + "contour-veg.png")
    # cv2.imwrite(out_path + "contour-veg.png", contour_image_veg)
    # cv2.imwrite(out_path + "contour-ter.png", contour_image_ter)
    
    # plt.imshow(contour_image)

    total_area_veg = ((sum(cv2.contourArea(contour) for contour in contours_veg))/(640*640))*100
    total_area_ter = ((sum(cv2.contourArea(contour) for contour in contours_terrain))/(640*640))*100
    
    if prt == True:
        write_text_opencv(contour_image_veg,str(total_area_veg), out_path + "contour-veg.png")
        write_text_opencv(contour_image_ter,str(total_area_ter), out_path + "contour-ter.png")

    # print("Total area of vegetation patches:", total_area_veg,"Total area of terrain patches:", total_area_ter)
    return total_area_veg, total_area_ter

# Display the binary mask
# plt.imshow(binary_mask)
# calc_area_n(mask_path = '/content/DeepLabV3Plus-Pytorch/test_results/5.png')



def calc_area(mask_path):
# Load the segmentation mask
    mask = cv2.imread(mask_path)

    lower_green = np.array([0, 200, 0])
    upper_green = np.array([50, 255, 50])
    
    green_mask = cv2.inRange(mask, lower_green, upper_green)
    mask[green_mask > 0] = [0, 255, 0]
    
    binary_mask = np.all(mask == [0, 255, 0], axis=-1).astype(np.uint8)
    # plt.imshow(binary_mask)
    cv2.imwrite('mask.png', binary_mask)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_image = mask.copy()
    # cv2.drawContours(contour_image, contours, -1, (80, 80, 80), 2)
    total_area = sum(cv2.contourArea(contour) for contour in contours)

    # print("Total area of green patches:", total_area)
    return total_area

# Display the binary mask
# plt.imshow(binary_mask)
# calc_area(mask_path = 'D:/RLHS-MITACS/Data/trials/2a.png_mask.jpg')