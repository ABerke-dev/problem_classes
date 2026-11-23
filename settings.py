from pathlib import Path

MODEL = "nvidia/nemotron-3.5-super-vl:free"     # describer model, paid slug WITHOUT ':free'
JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl:free"            # model that labels the relations (text only)
IMAGE_DIR = Path("images")       # for a small pilot use Path("images/Agnieszka Dziadek")
DB_FILE = "images_nvidia_nemotron-3.5-super-vl_free.db"

CATEGORIES = {
    "form_geometry": "shape, proportions, scale, volume, lines",
    "composition": "arrangement, rhythm, symmetry, balance, hierarchy, focal point",
    "visual_material": "colour, texture, materials, patterns, lighting",
    "function_ergonomics": "usability, accessibility, interaction, user needs",
    "context_meaning": "style, cultural references, symbolism, identity, narrative",
    "structure_technology": "construction, assembly, technical solutions",
}