import json

import sklearn.linear_model
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import numpy as np
from hyperopt import hp
import pickle
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import train_test_split
from sklearn import datasets
from fastapi.responses import JSONResponse
import asyncio
import random
from fastapi.responses import FileResponse

async def background_process(data: Dict):
    # Здесь может быть ваш фоновый процесс
    await asyncio.sleep(5)  # Пример: ждем 5 секунд

    print("Background process finished")

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

xgb_reg_params = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
xgb_fit_params = {
    'eval_metric': 'rmse',
    'early_stopping_rounds': 10,
    'verbose': False
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
class HPOpt(object):

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}
def find_delimiter(path):
    sniffer = csv.Sniffer()
    with open(path) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter

def preprocessing(file_name, target):
    set_col = dict()
    sep = find_delimiter(f'{file_name}')
    #print("----------", sep, "----------------")
    df = pd.read_csv(f'{file_name}', sep=sep)
    if 'id' in df.columns[0]:
        df = df.drop([df.columns[0]], axis=1)


    col_prep = {}
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            print(f"{col} - это числовой столбец.")
        else:
            try:
                df[col] = pd.to_datetime(df[col])
                df.drop(col, axis=1)
                col_prep[col] = 'del'
            except:
                print(df[col].dtype)
                # Применяем one-hot encoding
                df_encoded = pd.get_dummies(df, columns=[col])

                col_prep[col] = [f"{col}_{i}" for i in pd.unique(df[col])]


    print(col_prep)





    a = dict(df.dtypes)
    change_col = []

    for key, value in a.items():
        if str(value) not in ["int64", "float64"]:
            change_col.append(key)

    if len(change_col) > 0:
        for i in change_col:
            d = dict()
            mass = df[i].unique()
            for index, key in enumerate(mass):
                d[key] = index
            df[i] = df[i].map(d)
            set_col[i] = d

    y = df[target]
    x = df.drop([target], axis=1)
    # print(x)
    return x, y, set_col


def test_model(file_name,X_test,y_test):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    # Преобразуйте модель обратно в формат XGBoost
    #xgb_model = LinearRegression()
    #xgb_model._Booster = model.get_booster()

    # Теперь у вас есть модель XGBoost, которую можно использовать для тестирования
    y_pred = model.predict(X_test)

    # Оцените качество модели, например, используя среднюю квадратичную ошибку
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", model.score(X_test,y_test))
def train_model(file_name, target):

    x, y, set_col = preprocessing(file_name, target)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    obj = HPOpt(X_train, X_test, y_train, y_test)
    xgb_opt = obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
    optimal_xgb_model = xgb.XGBRegressor(
        #colsample_bytree=xgb_opt[0]['colsample_bytree'],
        #learning_rate=xgb_opt[0]['learning_rate'],
        #max_depth=xgb_opt[0]['max_depth'],
        #min_child_weight=xgb_opt[0]['min_child_weight'],
        #subsample=xgb_opt[0]['subsample'],
    )
    print(xgb_opt)
    print("-------", X_train)
    print("-------", y_train)
    #xbg_model = sklearn.linear_model.LinearRegression()
    #xbg_model.fit(X_train, y_train)
    optimal_xgb_model.fit(X_train, y_train)
    print("check2")
    #(xbg_model.score(X_test,y_test))

    number = random.randint(1, 350)
    pkl_filename = f"pickle_model_{number}.pkl"

    with open(pkl_filename, 'wb') as file:
        pickle.dump(optimal_xgb_model, file)
    #test_model('pickle_model.pkl',X_test,y_test)

    print(pkl_filename)
    return pkl_filename

app = FastAPI()

class Item(BaseModel):
    dataSetName: str
    trainInfo: dict


class Item_s(BaseModel):
    file: UploadFile = File(...)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    with open(file.filename, "wb") as f:
        f.write(file.file.read())


    return {"filename": file.filename + "12312312"}


@app.post("/use_model_file/")
async def use_model(file: UploadFile = File(...)):
    #text_data = item.text
    #file_contents = await item.file.read()

    # Логика обработки данных
    # ...
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    return {"asd" : "vlaaaaaaaaaaaaad"}


@app.post("/use_model_text/")
async def use_model(data: Dict):
    # Ваша логика обработки данных
    model_name = data['model']
    file_name = data['file_name']

    print(data)
    # Загрузка модели из файла .pkl
    # Загрузка модели из файла .pkl
    print(type(model_name), model_name)
    #model_name = json.loads(model_name)
    print(type(model_name), model_name)
    with open(model_name, 'rb') as file:
        loaded_model = pickle.load(file)
    print(file_name)
    df = pd.read_csv(file_name)
    print("123")
    # Теперь вы можете использовать загруженную модель для предсказаний
    # Например, если у вас есть тестовые данные X_test:
    # X_test = [[1, 2, 3], [4, 5, 6]]  # Замените это на свои тестовые данные
    predictions = loaded_model.predict(df)
    print(predictions)

    return {"predictions": str(predictions)}



@app.put("/create-item/")
async def create_item(item: Item, background_tasks: BackgroundTasks):
    # Теперь переменная item - это экземпляр класса Item
    # и её структура соответствует ожидаемой структуре JSON-запроса

    print(item)
    #asyncio.create_task(train_model1(item.dataSetName, item.trainInfo['target']))
    filename = train_model(item.dataSetName, item.trainInfo['target'])
    #background_tasks.add_task(background_train_model, item.dataSetName, item.trainInfo['target'])
    response_data = {"message": "Item created successfully", "item": item}

    return {'file_name' : filename}

@app.get("/downloadfile/{filename}")
async def download_file(filename: str):
    file_path = f"{filename}"  # Замените на путь к вашему файлу
    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="26.88.188.99", port=5014)