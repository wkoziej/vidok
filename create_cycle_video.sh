#!/bin/bash

# Skrypt do tworzenia filmu z efektem cyklu (do przodu i do tyłu)
# Użycie: ./create_cycle_video.sh <nazwa_pliku_wejściowego>

# Sprawdzenie czy podano argument
if [ $# -eq 0 ]; then
    echo "Błąd: Nie podano nazwy pliku wejściowego"
    echo "Użycie: $0 <nazwa_pliku_wejściowego>"
    echo "Przykład: $0 output_250524_033913_417_5907.mp4"
    exit 1
fi

# Przypisanie argumentu do zmiennej
INPUT_FILE="$1"

# Sprawdzenie czy plik istnieje
if [ ! -f "$INPUT_FILE" ]; then
    echo "Błąd: Plik '$INPUT_FILE' nie istnieje"
    exit 1
fi

# Wygenerowanie nazwy pliku wyjściowego
# Usuwa rozszerzenie i dodaje prefiks "cycle_"
BASENAME=$(basename "$INPUT_FILE" .mp4)
OUTPUT_FILE="cycle_${BASENAME}.mp4"

# Informacja o przetwarzaniu
echo "Przetwarzanie pliku: $INPUT_FILE"
echo "Plik wyjściowy: $OUTPUT_FILE"
echo "Tworzenie filmu z efektem cyklu..."

# Wykonanie komendy ffmpeg
ffmpeg -i "$INPUT_FILE" -filter_complex "[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0" "$OUTPUT_FILE"

# Sprawdzenie czy operacja się powiodła
if [ $? -eq 0 ]; then
    echo "✓ Sukces! Film z efektem cyklu został utworzony: $OUTPUT_FILE"
else
    echo "✗ Błąd podczas przetwarzania filmu"
    exit 1
fi 