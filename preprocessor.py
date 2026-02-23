import cv2
import numpy as np
from pathlib import Path

class ODIRImageProcessor:
    """Simple processor for ODIR retinal images"""
    
    def __init__(self, output_size=512):
        self.output_size = output_size
    
    def load_image(self, input_path,gamma=0.9, threshold=10):
        img = cv2.imread(str(input_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        img = self.preprocess_image(img,threshold=threshold,gamma=gamma)
        return img

    def process(self, input_path, output_path=None, skip_if_exist=True, gamma=0.9, threshold=10):
        """
        Process a single ODIR image
        
        Args:
            input_path: Path to input image
            output_path: Optional path to save result. If None, returns image array
        
        Returns:
            Processed image as numpy array
        """
        if skip_if_exist and output_path is not None:
            if os.path.exists(output_path):
                print(f"Skip as processed image existed: {output_path}")
                return

        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Call Preprocess script
        img = self.preprocess_image(img,threshold=threshold,gamma=gamma)

        # Convert back to BGR for saving/display
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(str(output_path), img_bgr)
            print(f"Saved processed image to: {output_path}")
        
       # return img_bgr
    def preprocess_image(self, img, threshold=10, gamma=0.9):
        img = self.resize(img)
        img = self.crop_black_border(img, thresh=threshold)
        img = self.crop_fundus_circle(img)
        img = self.center_retina(img)
        img = self.mask_outside_retina(img)
        img = self.gamma_correction(img, gamma=gamma)
        return img

    def gamma_correction(self, img, gamma=0.9):
        img_float = img.astype(np.float32) / 255.0
        # Gamma correction
        img_gamma = np.power(img_float, gamma)
        # Convert back to 0–255 for saving
        img_result = (img_gamma * 255).astype(np.uint8)
        return img_result

    def resize(self, img):
        return cv2.resize(img, 
                          (self.output_size, self.output_size), 
                          interpolation=cv2.INTER_CUBIC)

    def crop_black_border(self, img, thresh=10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # mask of non-black pixels
        mask = gray > thresh

        if not np.any(mask):
            return img  # fallback safety

        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return img[y_min:y_max+1, x_min:x_max+1]

    def crop_fundus_circle(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)
        return img[y:y+h, x:x+w]

    def mask_outside_retina(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return cv2.bitwise_and(img, mask)

    def center_retina(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]

        # pad to square
        h, w = crop.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=crop.dtype)

        y0 = (size - h) // 2
        x0 = (size - w) // 2
        padded[y0:y0+h, x0:x0+w] = crop

        return padded


    def apply_clahe(self, image):
        """
            Apply CLAHE contrast enhancement
            This is a bit controversial as some suggested it, while others said it make it worse
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge back
        lab = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def process_folder(self, input_folder, output_folder, skip_if_exist, extension=".jpg"):
        """
        Process all images in a folder
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_folder.glob(f"*{extension}"))
        print(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            output_path = output_folder / img_path.name
            try:
                self.process(img_path, output_path, skip_if_exist=skip_if_exist)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

# Initialize processor
#processor = ODIRImageProcessor(output_size=512)

# Process single image
#processor.process(
#    input_path="path/to/your/image.jpg",
#    output_path="path/to/save/processed_image.jpg"
#)

# OR process entire folder
# processor.process_folder(
#     input_folder="path/to/images",
#     output_folder="path/to/processed_images"
# )