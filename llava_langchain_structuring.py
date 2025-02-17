from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from PIL import Image
import torch

# Load LLAVA Model
llava_model_name = "liuhaotian/llava-v1.5-7b"
processor = LlavaProcessor.from_pretrained(llava_model_name)
model = LlavaForConditionalGeneration.from_pretrained(llava_model_name, torch_dtype=torch.float16, device_map="auto")

# Load Free LLM (LLaMA 2 or Mistral)
llm_model_name = "meta-llama/Llama-2-7b-chat-hf"  # Alternative: "mistralai/Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model e_map="auto")

# LangChain LLM Pipeline
llm = HuggingFacePipeline(model=llm_model, tokenizer=tokenizer)

# Function to Extract Text from Images Using LLAVA
def extract_text_from_image(image_path, prompt="Extract medical text from this report"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    generated_ids = model.generate(**inputs, max_new_tokens=300)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extracted_text

# LangChain Prompt for Structuring Medical Data
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Given the extracted medical text below, convert it into structured JSON format with key values such as:
    - Patient Name
    - Report Date
    - Hemoglobin (g/dL)
    - RBC Count (million/uL)
    - WBC Count (cells/uL)
    - Platelet Count (cells/uL)
    - AI Recommendations

    Medical Text: {text}
    """
)

chain = LLMChain(llm=llm, prompt=prompt)

# Example Usage
image_path = "blood_report.png"
raw_text = extract_text_from_image(image_path)
structured_output = chain.run(raw_text)

print("Structured Medical Report:\n", structured_output)
= AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.float16, devic