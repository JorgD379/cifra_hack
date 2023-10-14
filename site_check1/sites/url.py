from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("register", views.registration_view, name="registration_view"),
    path('login', views.login_view, name='login_view'),
    path('registration_success/', views.registration_success, name='registration_success'),
    path('profile', views.profile_view, name='profile'),
    path('create_something', views.create_something, name='create_something'),
    # URL и представление для создания чего-то
    path('logout', views.logout_view, name='logout'),  # URL и представление для выхода
    path('upload-csv/', views.upload_csv, name='upload_csv'),
    path('mymodel/', views.mymodels, name='mymodel'),
    path('train_csv/', views.train_csv, name='train_csv'),
    path('use_mod', views.use_mod, name='use_mod'),
    path('offer', views.offer, name='offer'),
    path('pay', views.pay, name='pay'),
    # path('upload-zip/', views.upload_zip, name='upload_zip'),
]
