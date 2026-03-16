#!/bin/bash

# Название выходного файла
OUTPUT="project_context.txt"

# Очищаем старый файл
> "$OUTPUT"

echo "Сборка контекста в $OUTPUT..."

# 1. Собираем список всех нужных файлов (сортируем по алфавиту)
# Игнорируем скрытые папки (.git), кэш (__pycache__), и папки с результатами генерации (agents, landscape)
FILES=$(find . -type f \
    \( -name "*.py" -o -name "*.yaml" -o -name "*.md" -o -name "*.txt" -o -name "*.sh" \) \
    -not -path '*/\.*' \
    -not -path '*/__pycache__/*' \
    -not -path './landscape/*' \
    -not -path './agents/*' \
    -not -path './venv/*' \
    -not -name "$OUTPUT" \
    -not -name "collect*.sh" | sort)

# 2. Создаем "Оглавление" для LLM
echo "================================================================================" >> "$OUTPUT"
echo "PROJECT ARCHITECTURE (INDEX)" >> "$OUTPUT"
echo "================================================================================" >> "$OUTPUT"
echo "$FILES" >> "$OUTPUT"
echo -e "\n\n" >> "$OUTPUT"

# 3. Записываем содержимое каждого файла
for file in $FILES; do
    echo "Добавляю: $file"
    
    echo "================================================================================" >> "$OUTPUT"
    echo "FILE: $file" >> "$OUTPUT"
    echo "================================================================================" >> "$OUTPUT"
    
    cat "$file" >> "$OUTPUT"
    
    echo -e "\n\n" >> "$OUTPUT"
done

echo "--------------------------------------------------"
echo "Готово! Весь код собран в файл: $OUTPUT"
echo "Всего строк: $(wc -l < "$OUTPUT")"