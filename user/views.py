from django.shortcuts import render


# Create your views here.
def user_page(request, *args, **kwargs):
    return render(request, "userpage.html", {})
