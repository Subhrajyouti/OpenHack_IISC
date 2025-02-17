from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate

# Load the PDF
pdf_path = 'blood_report.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the text into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create an LLM for processing the extracted text (Here we're using OpenAI, you can replace with another LLM or a multimodal LLM)
llm = OpenAI(temperature=0)

# You can define a prompt template for extracting specific details like blood report parameters
template = """
Given the following blood report, extract the relevant parameters such as RBC, WBC, Hemoglobin, etc.:
{document}
"""
prompt = PromptTemplate(input_variables=["document"], template=template)

# Build the chain for processing the document using LangChain's AnalyzeDocumentChain
chain = AnalyzeDocumentChain(combine_docs_chain=llm)

# Process the text chunks
results = []
for text in texts:
    result = chain.run(input_document=text)
    results.append(result)

# Combine the results
extracted_data = "\n".join(results)

# Output the extracted data
print(extracted_data)
