import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from synthesize import synthesize_image, pil_to_np
from PIL import Image


def load_and_preprocess_images(folder_path, target_size=(32, 32), normalize=True, test_size=0.2):

    synthetic_images = []
    clean_images = []

    i = 0
    for filename in os.listdir(folder_path):
        # print(filename)
        image_path = os.path.join(folder_path, filename)
        # image = cv2.imread(image_path)
        image = Image.open(image_path).convert('RGB')
        # image = cv2.resize(image, target_size)
        image = image.resize(target_size)

        # Convert BGR image to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if normalize:
        #     image = image.astype("float32") / 255.0

        # clean_images.append(image)

        synthetic_image = synthesize_image(image)
        # synthetic_images.append(synthetic_image)

        # image.show('clean')
        # synthetic_image.show('synthetic')
        # input("Press any key to continue...")
        # image.close()
        # synthetic_image.close()

        img = cv2.cvtColor(np.transpose(pil_to_np(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
        syn_img = cv2.cvtColor(np.transpose(pil_to_np(synthetic_image), (1, 2, 0)), cv2.COLOR_RGB2BGR)

        clean_images.append(img)
        synthetic_images.append(syn_img)

        # cv2.imshow('image', img)
        # cv2.imshow('synthetic image', syn_img)
        # cv2.waitKey(0)
        i+=1
        if i == 100:
            break

    clean_images = np.array(clean_images)
    synthetic_images = np.array(synthetic_images)

    X_train, X_val, y_train, y_val = train_test_split(synthetic_images, clean_images, test_size=test_size)

    # X_train2, X_val2 = train_test_split(clean_images, test_size=test_size)

    return (X_train, X_val), (y_train, y_val)


if __name__ == "__main__":

    # Example usage
    folder_path = "data/Flickr2K"
    data1, data2 = load_and_preprocess_images(folder_path)