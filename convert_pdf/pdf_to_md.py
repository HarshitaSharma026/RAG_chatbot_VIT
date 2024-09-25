import nest_asyncio
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse


load_dotenv()
nest_asyncio.apply()

doc = LlamaParse(result_type="markdown").load_data("./docs/Mtech_curriculum/information_security.pdf")

filename = "information_security.md"
with open(filename, 'w') as file:
    file.write(doc[0].text)