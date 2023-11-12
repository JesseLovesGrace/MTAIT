import nltk
import pdfplumber
import io
from PIL import Image
import pytesseract
from transformers import MarianMTModel, MarianTokenizer

nltk.download("punkt")


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page_num, page in enumerate(pdf.pages, 1):
            # Attempt to extract text using pdfplumber
            text += page.extract_text()

            # If no meaningful text is extracted, try OCR on images
            if not text.strip() and page.images:
                for i, image in enumerate(page.images, 1):
                    print(f"Debug: Page {page_num}, Image {i} structure: {image}")
                    # Modify the next line based on the actual structure of page.images
                    img_data = image.get("image_data", None)
                    if img_data:
                        img = Image.open(io.BytesIO(img_data))
                        ocr_result = pytesseract.image_to_string(img, lang="eng")
                        text += ocr_result
    return text


def split_into_chunks(text, max_chunk_length=500):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def translate_chinese_to_english(chinese_text):
    model_name = "Helsinki-NLP/opus-mt-zh-en"  # Chinese to English translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Split the input text into chunks
    chunks = split_into_chunks(chinese_text, max_chunk_length=500)

    # Translate each chunk and concatenate the results
    english_chunks = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1} of {total_chunks} ({((i + 1) / total_chunks) * 100:.2f}% complete)")
        input_ids = tokenizer.encode(chunk, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(input_ids)
        english_chunk = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        english_chunks.append(english_chunk)

    english_text = " ".join(english_chunks)
    return english_text


def save_text_to_file(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


# Input and output file paths
input_pdf_path = "C:\\Users\\jesse\\Desktop\\MT\\Mo_Yan\\Mo_Yan_Copy.pdf"
output_file_path = "C:\\Users\\jesse\\Desktop\\MT\\Mo_Yan\\Mo_Yan_Copy_MT.txt"

# Extract text from PDF
chinese_text = extract_text_from_pdf(input_pdf_path)

# Translate Chinese to English
english_text = translate_chinese_to_english(chinese_text)

# Save translated text to a new file
save_text_to_file(english_text, output_file_path)

print(f"\nTranslation saved to: {output_file_path}")
