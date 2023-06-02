## Репозиторий **Middle DS School: Text Recognition**

### О репозитории
Модуль **Text Recognition** посвящен технологиям **AI** в области документооборота, а 
именно задачам распознавания, классификации и извлечения информации из неструктурированного контента. 
Современные методы **CV**, **NLP** и мультимодальные подходы позволяют полностью 
оцифровывать любой документ, извлекать таблицы, рукописные элементы, а также необходимые и сущности и 
факты практически без разметки данных (**zero-shot** парадигма).

В данном репозитория представлен полный цикл, от подготовки датасета и создания 
синтетических данных, до тестирования готовых моделей и анализа ошибок. 
В данном репозитории находятся следующие модели и архитектуры: 
- сегментация изображений
- классификация изображений
- детекция объектов на изображении
- метрическое обучение
- задача NER
- текстовая классификация

Также будут рассмотрены инструменты для создания процесса **MLOPs**, а именно:
**W&B**, фреимворки для подготовки сервисов (**onnx** и **openvino**). На выходе 
получен готовый сервис с полным пайплайном обработки документа, начиная с 
распознавания текста на изображении и заканчивая постобработкой 
результатов работы **NER** моделей для составление финального структурированного ответа.

---

### Из чего состоит репозиторий

####  Notebooks

Папка с ноутбуками

В каждом ноутбуке решается определенная задача

#####  a_Text_Detection

Первая часть приложения -- модель, которая детектирует строки текста на изображении.
В качестве модели используется модель DB [статья](https://arxiv.org/abs/1911.08947), у которой есть условные "тело" (или "кодировщик") и "голова" (или "декодировщик").


