"""
Config script
"""
import os
from dotenv import load_dotenv
from numpy import uint8, uint16, float16

dotenv_path: str = '.env'
load_dotenv(dotenv_path=dotenv_path)

MAX_COLUMNS: int = int(os.getenv('MAX_COLUMNS'))
WIDTH: int = int(os.getenv('WIDTH'))
CHUNK_SIZE: uint16 = uint16(os.getenv('CHUNK_SIZE'))
PALETTE: str = os.getenv('PALETTE')
FONT_SIZE: uint8 = uint8(os.getenv('FONT_SIZE'))
ENCODING: str = os.getenv('ENCODING')
RE_PATTERN: str = os.getenv('RE_PATTERN')
RE_REPL: str = os.getenv('RE_REPL')

COLORS: list[str] = ["lightskyblue","coral","palegreen"]
FIG_SIZE: tuple[uint8, uint8] = (15, 8)
DTYPES: dict = {
	'resultado_diagnóstico': str, 'radio': 'uint8', 'textura': 'uint8',
	'perímetro': 'uint16', 'área': 'uint16', 'suavidad': 'float16',
	'compacidad': 'float16', 'simetría': 'float16',
	'dimensión_fractal': 'float16'}
converters: dict = {
	'TotalCharges': lambda x: float16(x.replace(' ', '0.0'))}
NUMERICS: list[str] = [
	'uint8', 'uint16', 'uint32', 'uint64',
	'int8', 'int16', 'int32',
	'int64',
	'float16', 'float32', 'float64']
RANGES: list[tuple] = [
	(0, 255), (0, 65535), (0, 4294967295), (0, 18446744073709551615),
	(-128, 127), (-32768, 32767), (-2147483648, 2147483647),
	(-18446744073709551616, 18446744073709551615)]
