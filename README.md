# Семантика понятия «народ» в латыни: от Античности к Средневековью

**Репозиторий к статье:** *Кузнецов А.В. "Семантика понятия «народ» в латыни: количественный анализ перехода от классической античности к высокой схоластике с помощью моделей word embeddings"*

**Автор:** Алексей Кузнецов, к.и.н., Институт всеобщей истории РАН


---

## О проекте

Данный репозиторий содержит код и материалы, использованные в исследовании семантического сдвига латинской лексики, обозначающей понятие «народ» (*populus, plebs, gens, natio, vulgus*), при переходе от классической античности к высокой схоластике.

В исследовании применяется метод дистрибутивной семантики (word embeddings) для сравнительного анализа двух семантических моделей:

1.  **Opera Latina**: модель, обученная на корпусе текстов классических римских авторов.
2.  **Opera Maiora**: модель, обученная на корпусе сочинений Фомы Аквинского (XIII в.).

Основной вывод работы заключается в том, что был зафиксирован не просто сдвиг в значениях отдельных слов, а фундаментальная смена самого принципа организации концептуального поля — от семантики, основанной на конкретной социально-правовой практике Рима, к семантике, выстроенной по абстрактной теолого-философской схеме Средневековья.

## Структура репозитория

-   `/article`: содержит финальную версию научной статьи в формате PDF.
-   `/notebooks`: содержит Jupyter/Google Colab ноутбук с кодом для проведения анализа и визуализации.
-   `requirements.txt`: список необходимых Python-библиотек для воспроизведения результатов.
-   `LICENSE`: лицензия на использование материалов репозитория.

## Воспроизведение результатов

Для воспроизведения анализа и визуализаций, представленных в статье, необходимо выполнить следующие шаги:

### 1. Клонирование репозитория

```bash
git clone https://github.com/ВАШ_ЛОГИН/latin-historical-semantics.git
cd latin-historical-semantics
```

### 2. Установка зависимостей

Рекомендуется создать виртуальное окружение для установки необходимых библиотек.

```bash
python -m venv venv
source venv/bin/activate  # Для macOS/Linux
# venv\Scripts\activate    # Для Windows

pip install -r requirements.txt
```

### 3. Запуск анализа

Откройте и запустите Jupyter Notebook `notebooks/analysis_and_visualization.ipynb`. Ноутбук содержит все этапы анализа:
- Загрузку предобученных моделей с сервера LiLa Project.
- Количественный анализ (вычисление косинусного сходства).
- Снижение размерности с помощью PCA.
- Построение итоговых графиков семантических пространств.

**Примечание:** Для запуска ноутбука не требуется скачивать модели вручную, скрипт загрузит их автоматически. Требуется стабильное интернет-соединение.

## Цитирование

Если вы используете материалы данного исследования в своей работе, пожалуйста, цитируйте статью:

> Кузнецов А.В. (2025). Семантика понятия «народ» в латыни: количественный анализ перехода от классической античности к высокой схоластике с помощью моделей word embeddings. *[Название журнала/сборника, том, страницы]*.

## Лицензия

Данный проект лицензирован под [MIT License](LICENSE) - см. файл LICENSE для подробностей.

MIT License

Copyright (c) 2025 Alexey Kuznetsov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.