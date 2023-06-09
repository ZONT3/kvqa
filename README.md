# Kapusta VQA

## Цели проекта

- Разработка системы оповещения о чрезвычайных ситуациях и их предотвращения
- Разработка модели VQA с использованием state-of-the-art технологий
- Интеграция модели в систему в качестве ее ядра

## Этот репозиторий

Код модели VQA и ее API

## Авторы

- Вейценфельд Д. А.
- Научный руководитель: Киселев Г. А.

Все вышеперечисленные относятся к:
1. Российский Университет Дружбы Народов,
   Факультет Физико-Математических и Естественных Наук,
   Кафедра Информационных Технологий
2. Институт Системного Анализа ФИЦ ИУ РАН

## Описание системы

**Этот раздел на данный момент не полностью соответствует действительности, и ожидает переработки**

Основной модуль - VQA (vqa) содержит подмодули kvqa_trainval (далее - trainval) и kvqa_dataset.

В подмодуле trainval используется класс Dataset из kvqa_dataset для подготовки и получения данных и дальнейшего 
обучения модели на них.

В нем же используется модуль modeling, содержащий конечный KVQAModel.

Этот класс - модель VQA, состоящая из модуля текста (KVQAText) и кросс-модальности текста со зрением (KVQAXModal)

KVQAModel обучается и используется для генерирования ответов в trainval.

Данные, передаваемые из Dataset в KVQAModel, являются парами вопросов ("сырых", в виде строки) и уже закодированных 
изображений.

Ответ, генерируемый KVQAModel является меткой ответа из набора данных (label). Для конвертации из метки в строку 
(ответ на естественном языке) используется метод класса Dataset - label_to_answer(label)

Перед любой из задач (обучение, генерация ответа - train, val (evaluation) ), происходит подготовка набора 
данных - кеширование ответов в метки, а так же кодирование изображений. Вся эта подготовка реализована в 
подмодуле kvqa_dataset.

Кодирование изображений происходит в том случае, если обнаружены изображения в наборе данных, которые 
еще не были закодированы (закодированные изображения сохраняются на диск)

Кодирование осуществляется классом FeatureExtractor, который находится в модуле modeling. Этот класс 
использует класс KVQAVision - модуль зрения, который отделен от остальных двух (текста и кросс-модальности)

