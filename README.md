# Vidok - FramePack Video Generation

Projekt do automatycznego generowania wideo z obrazków przy użyciu AI.

## 🎬 Przeznaczenie

**Vidok** to narzędzie do tworzenia krótkich filmów animowanych z pojedynczych zdjęć. Wykorzystuje model FramePack (HunyuanVideo) do generowania realistycznych ruchów i animacji na podstawie promptów tekstowych.

## 🚀 Główne funkcje

- **Generowanie wideo z obrazków** - przekształca statyczne zdjęcia w animowane filmy
- **Prompty tekstowe** - opisujesz jaki ruch chcesz zobaczyć (np. "osoba tańczy gracefully")
- **Tryb CLI** - automatyzacja i batch processing
- **Tryb Gradio** - interfejs webowy do interaktywnego użycia
- **Batch processing** - przetwarzanie wielu obrazków jednocześnie

## 📁 Struktura

- `FramePack/` - główny kod generowania wideo
- `prepare_prompts.py` - generowanie promptów dla obrazków (OpenAI Vision API)
- `batch_process.py` - przetwarzanie wsadowe wielu zadań
- `images*/` - katalogi z obrazkami do przetworzenia

## 🔄 Workflow

1. **Lokalnie**: Przygotuj obrazki i wygeneruj prompty
2. **Git**: Synchronizuj z serwerem
3. **Zdalnie**: Uruchom batch processing na GPU
4. **Rezultat**: Wygenerowane filmy MP4

## 💻 Użycie

### Interfejs webowy:
```bash
cd FramePack
python demo_gradio.py
```

### Tryb CLI:
```bash
cd FramePack  
python demo_gradio.py --cli -i image.jpg -p "person dances gracefully"
```

### Batch processing:
```bash
python batch_process.py --config configs/batch_001.json --parallel 2
```

## 🛠️ Wymagania

- CUDA-compatible GPU
- Python 3.8+
- Modele HunyuanVideo (pobierane automatycznie)
- OpenAI API key (dla generowania promptów)
