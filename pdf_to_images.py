from pdf2image import convert_from_path

def pdf_to_images(pdf_path):
    """Convert PDF pages to images."""
    images = convert_from_path(pdf_path)
    return images

# Example Usage
pdf_path = "blood_report.pdf"
images = pdf_to_images(pdf_path)

# Save the first page as an image for LLAVA
image_path = "page1.png"
images[0].save(image_path, "PNG")
print("Saved first page as:", image_path)
