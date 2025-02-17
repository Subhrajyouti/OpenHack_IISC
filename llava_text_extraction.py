from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load LLAVA Model
model_name = "liuhaotian/llava-v1.5-7b"
processor = LlavaProcessor.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def extract_text_from_image(image_path, prompt="Extract medical text from this report"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(**inputs, max_new_tokens=300)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extracted_text

# Example Usage
image_path = "blood_report.png"
print("Extracted Text:\n", extract_text_from_image(image_path))
