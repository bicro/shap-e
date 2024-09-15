import certifi
import ssl

context = ssl.create_default_context(cafile=certifi.where())