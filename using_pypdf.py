from pypdf import PdfReader
import os

pdf_dir = "./docs/Mtech_curriculum"
output_dir = "./docs/Mtech_curriculum"

# to check if the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# iterate over each file in the directory
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        # creating the path -> docs/Mtech_curriculum/<filename>
        pdf_path = os.path.join(pdf_dir, filename)
        reader = PdfReader(pdf_path)

        # opening a new txt file with the same name but different extension
        text_filename = filename.replace(".pdf", ".txt")
        text_filepath = os.path.join(output_dir, text_filename)

        # opening the pdf file and copying its text in newly created text file
        with open(text_filepath, "w", encoding="utf-8") as txt_file:
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    txt_file.write(text)
                    txt_file.write(f"\n")

print("Conversion for all files complete !!")
