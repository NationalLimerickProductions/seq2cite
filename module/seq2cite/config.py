from pathlib import Path


root = Path(__file__).parent.parent.parent
data = root / 'data'
raw = data / 'raw'
processed = data / 'processed'
final = data / 'final'
docs = root / 'docs'
models = root / 'models'
notebooks = root / 'notebooks'
src = root / 'src'

cord19_aws_bucket = 'ai2-semanticscholar-cord-19'

metadata_columns= [
    'cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
    'license', 'abstract', 'publish_time', 'authors', 'journal',
    'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse',
    'has_pmc_xml_parse', 'full_text_file', 'url'
]
