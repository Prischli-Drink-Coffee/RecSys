### Pipline

```
from pathlib import Path

# Создание и использование Splitter
splitter = SomeSplitterImplementation(
    drop_cold_users=True,
    drop_cold_items=True,
    query_column="user_id",
    item_column="item_id",
    timestamp_column="timestamp"
)

# Пример работы Splitter
interactions_df = ...  # Загрузка или создание DataFrame взаимодействий
(train, test), cold_users, cold_items = splitter.split(interactions_df)

# Сохранение результатов разбиения, если нужно
if cold_users is not None:
    cold_users.to_csv("cold_users.csv", index=False)

if cold_items is not None:
    cold_items.to_csv("cold_items.csv", index=False)

# Сохранение Splitter на диск
splitter_path = Path("saved_splitter")
save_splitter(splitter, splitter_path, overwrite=True)

# Загрузка Splitter с диска
loaded_splitter = load_splitter(splitter_path)

# Проверка: повторное разбиение с загруженным Splitter
(train_reloaded, test_reloaded), cold_users_reloaded, cold_items_reloaded = loaded_splitter.split(interactions_df)

# Убедимся, что результаты совпадают
assert train.equals(train_reloaded), "Train datasets do not match!"
assert test.equals(test_reloaded), "Test datasets do not match!"
if cold_users is not None:
    assert cold_users.equals(cold_users_reloaded), "Cold users do not match!"
if cold_items is not None:
    assert cold_items.equals(cold_items_reloaded), "Cold items do not match!"
```


### TorchDataset
