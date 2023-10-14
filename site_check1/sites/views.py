import csv
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from .forms import RegistrationForm, ChooseTargetForm, MyForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login, logout
from django.core.mail import send_mail
from .models import SiteUserProfile
from io import TextIOWrapper
import requests
import numpy as np
import csv
import json
import pandas as pd
import re


def registration_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('registration_success')
    else:
        form = RegistrationForm()
    return render(request, 'registration_form.html', {'form': form})


def login_view(request):
    print('BCBCBCBCBCBCBCBCBC')
    if request.method == 'POST':
        print('ABABBABABABBA')
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('profile')  # Перенаправление на страницу профиля или куда вам нужно
    else:
        form = AuthenticationForm()
    return render(request, 'enter.html', {'form': form})


def profile_view(request):
    # Добавьте здесь логику для страницы профиля
    return render(request, 'profile.html')


def registration_success(request):
    return render(request, 'profile.html')


def index(request):
    return render(request, 'base.html')


def create_something(request):
    # Ваша логика для создания чего-то
    return render(request, 'create.html')


def upload_csv(request):
    if request.method == 'POST':
        print('1')
        # print(form)
        print('2')
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        print('3')
        if csv_file:
            print('4')
            csv_data = []
            decoded_file = TextIOWrapper(csv_file.file, encoding='utf-8')
            reader = csv.reader(decoded_file)
            for row in reader:
                csv_data.append(row)
            name = fs.save(csv_file.name, csv_file)
            print(csv_data)
            df = pd.read_csv(f"./media/{name}", sep=find_delimiter(f"./media/{name}"))
            columns = df.columns
            col_form = []
            for col in columns:
                col_form.append((col, col))
            form = ChooseTargetForm(colums=col_form)
            print(form)
            return render(request, 'settings.html', {'data': csv_data, 'form': form, 'name': name})
    return render(request, 'create2.html')


def find_delimiter(path):
    sniffer = csv.Sniffer()
    with open(path) as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


def train_csv(request):
    print('1')
    name = request.GET.get("name")
    user = request.user  # Получаем текущего пользователя из запроса
    print(name)
    if request.method == 'POST':
        print("Зашел____________________________")
        form = ChooseTargetForm(request.POST)
        # if form.is_valid():
        print("Зашел1____________________________")
        df = pd.read_csv(f"./media/{name}", sep=find_delimiter(f"./media/{name}"))
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

        my_D = {}
        my_D['info'] = col_prep
        data = request.POST
        target = data.get('target')
        my_D['target'] = target
        print('-------------------')
        print(str(my_D))
        print('-------------------')
        files = {'file': (name, open(f"./media/{name}", 'rb'))}
        response = requests.post("http://26.88.188.99:5014/uploadfile/", files=files)
        print(response)
        param = {"dataSetName": name, "trainInfo": my_D}
        json_param = json.dumps(param)
        headers = {"Content-Type": "application/json"}
        r = requests.put('http://26.88.188.99:5014/create-item/', data=json_param, headers=headers)
        print(r.content)
        response_content = r.content.decode('utf-8').strip('"')
        # Создаем или обновляем запись SiteUserProfile для текущего пользователя
        user_profile = SiteUserProfile(user=user)
        user_profile.response_content = response_content
        user_profile.save()
        user = request.user
        user_profile = SiteUserProfile.objects.filter(user=user)
        response_content = [json.loads(profile.response_content)['file_name'] for profile in user_profile]
        context = {'data': name, 'names': response_content}
        subject = 'Обучена модель'
        message = 'Ваша модель обучена успешна!'
        try:
            send_mail(subject, message, 'nugaev.vlad@mail.ru', ['mo.17gruppa@mail.ru'])
        except Exception as e:
            print('Error sending email:', str(e))
        return render(request, 'mymodel.html', context)
    # Обработка ошибок, если необходимо
    return redirect('csv_content')

def logout_view(request):
    logout(request)
    return redirect('index')  #выход с профилем


def mymodels(request):
    user = request.user
    user_profile = SiteUserProfile.objects.filter(user=user)
    response_content = [json.loads(profile.response_content)['file_name'] for profile in user_profile]
    print(response_content)
    context = {'names': response_content}
    return render(request, 'mymodel.html', context)

def use_mod(request):
    file_name = request.GET.get("file_name")
    if request.method == 'POST':
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        if csv_file:
            name = fs.save(csv_file.name, csv_file)
            files = {'file': (name, open(f"./media/{name}", 'rb'))}
            user = request.user
            user_profile = SiteUserProfile.objects.filter(user=user)
            response_content = [profile.response_content for profile in user_profile]
            print(response_content)
            print()
            #res = json.loads(file_name)
            url = "http://26.88.188.99:5014/use_model_file/"
            print('----------------------------')
            #print(res)
            print('-------------------------')
            print('***************')
            print(name)
            print('***************')
            files = {'file': (name, open(f"./media/{name}", 'rb'))}
            response = requests.post(url, files=files)
            print(response.content)
            url = "http://26.88.188.99:5014/use_model_text/"
            data = {"model":  file_name,  "file_name": name}
            response = requests.post(url, json=data)
            print('--------------------------------------')
            print(response.content)
            # Разбор JSON-строки
            print(response.content.decode('utf-8'))
            data = json.loads(response.content.decode('utf-8'))
            # Извлечение чисел и создание массива
            predictions = np.fromstring(data['predictions'][1:-1], sep=' ')
            # Создание CSV-файла
            file_path = os.path.join(settings.MEDIA_ROOT, 'predictions.csv')
            # Создайте файл с предсказаниями и сохраните его
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Prediction'])
                for prediction in predictions:
                    writer.writerow([prediction])
            download_link = os.path.join(settings.MEDIA_URL, 'predictions.csv')
            print('--------------------------------------')
            return render(request, 'memod1.html', {'response_content': response_content, 'download_link': download_link})
    return render(request, 'memod.html')


def offer(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Получите данные из формы
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            email = form.cleaned_data['email']
            filename = form.cleaned_data['filename']
            # Отправьте письмо
            send_mail(
                'Новая заявка',
                f'Здравствуйте, {last_name} {first_name} мы рады, что вы стали заинтересованы нашим продуктом. Расскажите подробнее, что именно вас интересует по поводу услуги {filename}',
                'nugaev.vlad@mail.ru',  # Замените на ваш адрес электронной почты
                [f'{ email }'],  # Замените на адрес получателя
                fail_silently=False,
            )
            # Верните, например, страницу благодарности
            form = MyForm()
            return render(request, 'offer.html', {'form': form})
    else:
        form = MyForm()
        return render(request, 'offer.html', {'form': form})


def pay(request):
    return render(request, 'pay.html')
