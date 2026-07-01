import base64
import sys
data = "AAAAAAAAAAA="
decoded = base64.b64decode(data)
print(''.join(f'{byte:08b}' for byte in decoded))