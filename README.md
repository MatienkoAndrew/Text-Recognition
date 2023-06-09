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

#####  1. a_Text_Detection.ipynb

- Решена задача детекции строк текста на изображении.
- Построена модель DB [статья](https://arxiv.org/abs/1911.08947), у которой есть условные "тело" (или "кодировщик") и "голова" (или "декодировщик").

![image](https://github.com/MatienkoAndrew/MiddleDS/assets/29499863/3e5d604d-c87d-436d-87bf-993c9a343392)

#####  2. b_Optical_Character_Recognition.ipynb

- Решена задача OCR (Optical Character Recognition).
- Построена модель [CRNN (Convolutional Recurrent Neural Network)](https://arxiv.org/abs/1507.05717), которая умеет распознавать символы.

Архитектура модели CRNN:

![CRNN](https://images4.russianblogs.com/922/df/df7f964dc5a09b659096b55b705c96f2.png)

##### 3. c_Layout_Analisys.ipynb

- Решена задача получения связного текста из набора символов.
- Использована уже обученная модель из ноутбука 1, которая умеет находить строки, и на ее основе собираются тексты.

![image](https://github.com/MatienkoAndrew/MiddleDS/assets/29499863/b34aa6b0-fb7d-4b78-b5bb-be421755c8e1)


##### 4. d_Named_Entity_Recognition.ipynb

- Решена задача NER (Named Entity Recognition).
- Используется предобученная модель [BertForTokenClassification](https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/bert/modeling_bert.py#L1709).

![image](https://github.com/MatienkoAndrew/MiddleDS/assets/29499863/e6e8c316-7d34-4d52-89f8-f1fa10fcd0bc)

##### 5. e_Service_Assembly.ipynb

- Построен пайплайн объединения всех предыдущих моделей (detection -> OCR -> NER)
- Собран сервис на flask, почти:)

---

####  Resources

Здесь представлены утилиты для каждого ноутбука

---

####  models

Сохраненные модели:

- text_detection_model.jit - модель детекции текста;
- ocr_model.jit - модель OCR (по строке распознает символы);
- la.jit - модель для предсказания строк;
- paragraph_finder.pkl - алгоритм нахождения параграфа из строк;
- ner.jit - модель NER, распознавание именованных сущностей (***мб не сохранилась потому что много весит :)***.
