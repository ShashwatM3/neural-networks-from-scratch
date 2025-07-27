from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np
import random

# Load default font
font = ImageFont.load_default()

class ImagesSet:
    def __init__(self, iterations):
        self.iterations = iterations

    def distort_image(self, img):
        # Apply random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=255)

        # Apply random translation (shift)
        max_shift = 3  # pixels
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        img = ImageOps.expand(img, border=max_shift, fill=255)
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            fillcolor=255
        )
        img = img.crop((max_shift, max_shift, img.size[0] - max_shift, img.size[1] - max_shift))

        # Optionally add Gaussian blur a little
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Optionally add noise
        if random.random() < 0.3:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 10, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def initialize_images(self):
        flattened_images = []
        # Multiply dataset size by 5
        for j in range(self.iterations * 5):
            for i in range(1, 11):
                # Create a blank white 40x40 image
                img = Image.new('L', (40, 40), color=255)  # grayscale
                
                draw = ImageDraw.Draw(img)
                text = str(i)

                # Get text size
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Center text
                position = ((40 - text_width) // 2, (40 - text_height) // 2)
                draw.text(position, text, fill=0, font=font)  # black text

                # Distort the image
                img = self.distort_image(img)

                # Convert to numpy array and flatten
                arr = np.array(img)
                flattened = arr.flatten()
                flattened_images.append([flattened, [i]])

                # Optional: save distorted images for inspection
                # img.save(f"digit_{i}_var{j}.png")

        return flattened_images

    def convertImages(self, test_images):
        data = []
        for test_image in test_images:
            image_path = test_image[0][0]
            img = Image.open(image_path).convert('L')  # grayscale
            img = img.resize((40, 40), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            flat_array = img_array.flatten()

            data.append([flat_array, [test_image[1][0]]])
        return data
