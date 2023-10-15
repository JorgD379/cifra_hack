from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db import models





class RegistrationForm(UserCreationForm):
    email = forms.CharField(max_length=150, required=True)
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', 'email')


class ChooseTargetForm(forms.Form):
    def __init__(self, colums, *args, **kwargs):
        super(ChooseTargetForm, self).__init__(*args, **kwargs)
        self.fields['target'] = forms.ChoiceField(widget=forms.RadioSelect, choices=colums)


class MyForm(forms.Form):
    first_name = forms.CharField(label='Имя', max_length=100)
    last_name = forms.CharField(label='Фамилия', max_length=100)
    email = forms.CharField(label='Почта')
    filename = forms.ChoiceField(choices=[("Lite", "Lite"), ("Pro", "Pro"), ("Master", "Master")], label='Услуга')
