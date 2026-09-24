import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
from translator.translation import TranslationEngine

engine = TranslationEngine(models_root=Path("models"))
texts = [
    "Modern global enterprises produce high volumes of documentation in English, Hindi, and Bengali.",
    "Manual translation is costly, error-prone, and causes significant delays in operational workflows.",
    "The objective of this project is to build an automated, high-precision document translation engine powered by fine-tuned IndicTrans2 neural models.",
]

print("Starting batch translation with use_cache=True...")
t0 = time.time()
res = engine.translate_batch(texts, "English", "Hindi", batch_size=4)
t1 = time.time()
print(f"Batch completed in {t1-t0:.2f}s:")
for orig, tr in zip(texts, res):
    print(f"EN: {orig}")
    print(f"HI: {tr}\n")
