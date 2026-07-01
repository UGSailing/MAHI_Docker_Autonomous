import base64
import sys
data = "AAAAAAAAAAc="
decoded = base64.b64decode(data)
print(''.join(f'{byte:08b}' for byte in decoded))