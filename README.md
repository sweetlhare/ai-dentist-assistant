# ИИ-ассистент стоматолога
## Репозиторий с решением хакатона Цифровой Прорыв: сезон ИИ в УФО.
## Команда – Московские Зайцы 🐰

__Предсказания на тестовой выборке хранятся в файле test_predictions.xlsx__

__Ссылка на веб-приложение https://sweetlhare-ai-dentist-assistant-app-wv1dxb.streamlitapp.com/__

# 1. Запускаемость кода

## Веб-приложение

Весь необходимый код для веб-приложения, развернутого на Streamlit находится в папке app_src.

Чтобы запустить приложение локально, нужно установить зависимости из requirments.txt и вызвать в консоли 

streamlit run app.py

## Инференс

Пример инференса через код можно найти в __notebooks/yolo_1.ipynb__

# 2. Обоснованность выбранного метода (описание подходов к решению, их обоснование и релевантность задаче)

Предобработка данных осуществлена при помощи инструмента аннотации Supervisely. Часть разметки кариеса была переделана. Сделано это для того, чтобы исправить некачественные данные и разметку. Также было опробован метод псевдолейблинга, но не дал положительных результатов.

Основной моделью для детекции зубов и кариеса была выбрана архитектура YoloV5 в конфигурациях S и M. Данная модель является SOTA-методом для задач детекции и трекинга. Не тяжелые конфигурации были выбраны с целью сократить нагрузку на веб-сервис.

Для проработки нашей концепции с сегментацией данные о кариесе были размечены полигонами в Supervisely. После этого мы успели опробовать модель Unet, но не получили адекватных результатов. Однако, концепция сегментации в данной задаче имеет куда больше смысла, чем детекция. Но для реализации требуются более качественные данные. В будущем планируется использовать модели семейства Mask R-CNN.

Более подробно о результатах и планах по развитию проекта можно узнать из презентации.

# 3. Точность работы алгоритма (возможность оценить формальной метрикой с обоснованием выбора)

Accuracy ~ 76% (процент правильно классифицированных случаев есть/нет кариес)

# 4. Адаптивность / Масштабируемость

Наша разработка является максимально адаптивной и масштабируемой. Достигается это за счет использования веб-сервиса, который отметает необходимость в вычислительных мощностях на локальных устройствах пользователей. А также за счет того, что модели в сервисе явлюятся взаимозаменяемыми, так как изначально мы планировали заменить модель детекции на модель сегментации.

Мы прописали алгоритм развертывания приложения на других устройствах и подготовили файл необходимых зависимостей.

# 5. Отсутствие в решении импортного ПО и библиотек, кроме свободно распространяемого с обоснованием выбора

Решение полностью построено на open-source библиотеках:
- streamlit
- pytorch
- yolov5
- opencv

Помимо этого, хочу заметить, что сервис развернут на бесплатном сервере от Streamlit.

# 6. Наличие интеграционных интерфейсов, в первую очередь интерфейсов загрузки данных

Мы позаботились о том, чтобы пользователю было удобно загружать данные в веб-сервис. Можно загрузить как одно изображение, так и несколько (в случае отдельных фотографий верхней и нижней частей полости рта).

К тому же, пользователь видит информацию о качестве фотографий и, если качество недостаточно, он может сразу загрузить новое фото.

