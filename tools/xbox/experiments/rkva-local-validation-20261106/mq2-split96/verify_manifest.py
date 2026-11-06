from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
for line in (root/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1)
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('PASS manifest')
