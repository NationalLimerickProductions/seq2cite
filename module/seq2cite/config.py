from pathlib import Path


root = Path(__file__).parent.parent.parent
data = root / 'data'
raw = data / 'raw'
processed = data / 'processed'
docs = root / 'docs'
models = root / 'models'
notebooks = root / 'notebooks'
src = root / 'src'
